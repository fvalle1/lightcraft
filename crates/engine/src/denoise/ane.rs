//! The Neural Engine's side of AI Denoise (Apple silicon): the model as a Core ML network, set up like the graphics
//! card is — on a thread of its own under a crash marker and a time limit, checked against the CPU's answer on a
//! check tile and timed against it — and used while it keeps working.
//!
//! Core ML takes no ONNX file, so the installed model is rewritten as a Core ML model ([`lightcraft_denoise::coreml`]:
//! rescaled exactly so 16-bit arithmetic holds its activations, then written as a `.mlmodel`), compiled by Core ML and
//! kept in `<model folder>/coreml/` under a name that says which model file, tile and writer made it. Core ML decides
//! which layers the Neural Engine takes; a tile it gets wrong or cannot run is run on the CPU like a bad graphics
//! card's.

use super::gpu::{ANE_SETUP_MARKER, card_tile_ms, set_up_card};
use super::{Arc, AtomicUsize, Duration, Ordering, Path, PathBuf, RunError, SystemTime, TileRunner};
use lightcraft_coreml::{ComputeUnits, Model};
use lightcraft_denoise::coreml::{self, INPUT, OUTPUT};

/// Tiles run at once on the Neural Engine: one is being run while the next are packed and the last blended.
pub(super) const ANE_PARALLEL: usize = 2;

/// How far the Neural Engine's answer on the check tile may be from the CPU's (of the answer's largest value) before it is
/// not used. 16-bit arithmetic gives about 3×10⁻³ on RawNIND's model.
const ANE_AGREE: f32 = 2e-2;

/// Longest the Neural Engine may take to set up (measure, rewrite and compile the model, run the check tile and
/// time it) before the other devices do the work instead. Core ML compiles for the Neural Engine on first use.
const ANE_SETUP_LIMIT: Duration = Duration::from_secs(60);

/// Tiles the Neural Engine may get wrong (an error, numbers that are not finite) before it stops being used.
pub(super) const ANE_MAX_FALLBACKS: usize = 3;

/// Why the Neural Engine is not tried when a set-up marker is found.
pub(super) const ANE_CRASHED: &str = "LightCraft closed while it was setting up the Neural Engine for AI Denoise last time";

/// Why the Neural Engine is not used when its set-up ran out of time (the start of it).
pub(super) const ANE_SLOW: &str = "setting up the Neural Engine took longer than";

/// Why the Neural Engine is not used while a set-up that ran out of time is still going.
pub(super) const ANE_STUCK: &str = "the Neural Engine is still setting up from before (it took too long)";

/// The name of the Neural Engine in the status.
pub(super) const ANE_NAME: &str = "Apple Neural Engine";

/// A version of what is written, so a change to the writer never finds a model an older one made.
const WRITER_VERSION: u32 = 1;

/// Whether this computer can have a Neural Engine at all (Apple silicon, in a build with the `denoise` feature).
pub(super) const fn possible() -> bool {
    cfg!(all(target_os = "macos", target_arch = "aarch64"))
}

/// Whether `LIGHTCRAFT_GPU` (as read) turns the accelerators off for the process.
fn gpu_env_off(value: Option<&str>) -> bool {
    value.is_some_and(|v| matches!(v.trim(), "0" | "off" | "false" | "no"))
}

/// Why the user's switches keep the accelerators off, if they do: `LIGHTCRAFT_GPU=0` and
/// `LIGHTCRAFT_GPU_BACKEND=off` keep AI Denoise on the processor (docs/gpu-pipeline.md), the Neural Engine included.
pub(super) fn switched_off() -> Option<&'static str> {
    if gpu_env_off(std::env::var("LIGHTCRAFT_GPU").ok().as_deref()) {
        return Some("disabled by LIGHTCRAFT_GPU=0");
    }
    lightcraft_gpu::backend::compute_backends().is_none().then_some("disabled by LIGHTCRAFT_GPU_BACKEND=off")
}

/// Whether a set-up that did not work is worth trying again (a crash, the time limit), unlike one the model or the
/// computer cannot do.
pub(super) fn worth_retrying(why: &str) -> bool {
    why == ANE_CRASHED || why == ANE_STUCK || why.starts_with(ANE_SLOW)
}

/// The set-up's own words for the Neural Engine, from [`set_up_card`]'s for a graphics card.
fn for_ane(why: String) -> String {
    why.replace("the graphics card", "the Neural Engine")
        .replace("graphics card", "Neural Engine")
        .replace("the GPU runner", "the Neural Engine runner")
}

/// A model run through Core ML on the Neural Engine (with the processor for what it cannot take).
pub(super) struct AneRunner {
    model: Model,
    tile: usize,
    /// The network's input and output channels (4 and 3 for the Bayer-to-RGB models).
    channels: (usize, usize),
    /// The factor the model's answer is smaller by than the network's (a power of two, see `coreml::rescale`).
    output_scale: f32,
}

impl TileRunner for AneRunner {
    fn run(&self, input: &[f32]) -> Result<Vec<f32>, RunError> {
        let (t, (cin, cout)) = (self.tile, self.channels);
        let mut out = self.model.predict(&[1, cin, t, t], input).map_err(|e| RunError::Runtime(e.to_string()))?;
        if out.shape != [1, cout, 2 * t, 2 * t] {
            return Err(RunError::Runtime(format!("the Neural Engine's answer is shaped {:?}", out.shape)));
        }
        // back to the network's own scale (dividing by a power of two is exact)
        out.data.iter_mut().for_each(|v| *v /= self.output_scale);
        Ok(out.data)
    }
}

pub(super) struct AneSide {
    pub(super) runner: AneRunner,
    /// The check tile on the Neural Engine, the fastest of a few runs (ms).
    pub(super) ane_ms: f64,
    /// The check tile on the CPU, on one thread (ms).
    pub(super) cpu_ms: f64,
}

/// Where the Core ML model of the model file `onnx` (whose content hashes to `sha256`) for `tile` is kept.
fn cache_dir(onnx: &Path, sha256: &str, tile: usize) -> PathBuf {
    let short: String = sha256.chars().filter(char::is_ascii_hexdigit).take(16).collect();
    onnx.with_file_name("coreml").join(format!("{short}-t{tile}-v{WRITER_VERSION}"))
}

/// The scale a kept model was written with, if it is one this code could have written.
fn read_scale(dir: &Path) -> Option<f32> {
    let text = std::fs::read_to_string(dir.join("scale")).ok()?;
    let s: f32 = text.trim().parse().ok()?;
    // a power of two between 2^-120 and 1
    (s.is_finite() && s > 0.0 && s <= 1.0 && s.log2().fract() == 0.0 && s.log2() >= -120.0).then_some(s)
}

fn load(dir: &Path, tile: usize, channels: (usize, usize)) -> Result<AneRunner, String> {
    let output_scale = read_scale(dir).ok_or("the kept Core ML model has no valid scale")?;
    let model = Model::load(&dir.join("net.mlmodelc"), ComputeUnits::CpuAndNeuralEngine, INPUT, OUTPUT).map_err(|e| e.to_string())?;
    Ok(AneRunner { model, tile, channels, output_scale })
}

/// Folders of made-but-never-finished models beside `dir` (a crash, a power cut) go after an hour: a set-up that is
/// still running keeps touching its own.
fn remove_stale_parts(dir: &Path) {
    let (Some(parent), Some(name)) = (dir.parent(), dir.file_name().and_then(|n| n.to_str())) else { return };
    let Ok(entries) = std::fs::read_dir(parent) else { return };
    for e in entries.flatten() {
        let stale = e.file_name().to_str().is_some_and(|n| n.starts_with(&format!("{name}.part-")))
            && e.metadata()
                .and_then(|m| m.modified())
                .is_ok_and(|t| SystemTime::now().duration_since(t).is_ok_and(|age| age > Duration::from_secs(3600)));
        if stale {
            let _ = std::fs::remove_dir_all(e.path());
        }
    }
}

/// Measure how far the network's activations reach, rewrite it for 16-bit arithmetic, have Core ML compile it and keep
/// the result in `dir` (made beside it and moved into place, so a half-made one is never found).
fn build(net: &lightcraft_denoise::net::Net, tile: usize, dir: &Path) -> Result<(), String> {
    let runner = lightcraft_denoise::cpu::NetRunner::new(net, tile).map_err(|e| e.to_string())?;
    let peaks = runner.peaks(&lightcraft_denoise::runtime::check_tile(tile)).map_err(|e| e.to_string())?;
    let built = coreml::build(net, tile, &peaks).map_err(|e| e.to_string())?;
    // a folder of this set-up's own: another set-up of the same model (a second session, a retry while an earlier
    // one still runs after its time limit) never shares it
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    let name = dir.file_name().and_then(|n| n.to_str()).ok_or("the kept model has no name")?;
    let part = dir.with_file_name(format!("{name}.part-{}-{}", std::process::id(), NEXT.fetch_add(1, Ordering::Relaxed)));
    remove_stale_parts(dir);
    std::fs::create_dir_all(&part).map_err(|e| format!("could not make {}: {e}", part.display()))?;
    let result = (|| {
        let file = part.join("net.mlmodel");
        std::fs::write(&file, &built.mlmodel).map_err(|e| format!("could not write {}: {e}", file.display()))?;
        lightcraft_coreml::compile(&file, &part).map_err(|e| e.to_string())?;
        let _ = std::fs::remove_file(&file);
        std::fs::write(part.join("scale"), format!("{}\n", built.output_scale)).map_err(|e| format!("could not write the scale: {e}"))?;
        let _ = std::fs::remove_dir_all(dir);
        std::fs::rename(&part, dir).map_err(|e| format!("could not keep the Core ML model in {}: {e}", dir.display()))
    })();
    if result.is_err() {
        let _ = std::fs::remove_dir_all(&part);
    }
    result
}

/// Set the Neural Engine up for the model at `path`: the Core ML model made (or found kept) and loaded, the check tile
/// run on both and compared, both timed. `sha256` is the model file's.
pub(super) fn make_ane(cpu: &lightcraft_denoise::runtime::CpuRunner, path: &Path, sha256: &str, tile: usize) -> Result<AneSide, String> {
    if !possible() {
        return Err("this computer has no Neural Engine".into());
    }
    let net = cpu.net().clone();
    let channels = (net.in_channels, net.out_channels());
    // the check tile has the four planes of a Bayer mosaic
    if channels.0 != 4 {
        return Err(format!("the check tile has 4 channels, not the model's {}", channels.0));
    }
    let input = Arc::new(lightcraft_denoise::runtime::check_tile(tile));
    let on_ane = input.clone();
    let dir = cache_dir(path, sha256, tile);
    let ane = move || -> Result<_, String> {
        let runner = match load(&dir, tile, channels) {
            Ok(r) => r,
            Err(kept) => {
                // nothing kept, or what is kept does not load: make it again
                log::debug!("denoise: making the Core ML model ({kept})");
                build(&net, tile, &dir)?;
                load(&dir, tile, channels)?
            }
        };
        let got = runner.run(&on_ane).map_err(|e| format!("the Neural Engine could not run the check tile: {e}"))?;
        let ane_ms = card_tile_ms(&runner, &on_ane);
        Ok((runner, got, ane_ms))
    };
    let on_cpu = || {
        let started = web_time::Instant::now();
        (cpu.run(&input), started.elapsed().as_secs_f64() * 1000.0)
    };
    let ((runner, got, ane_ms), (want, cpu_ms)) =
        set_up_card(&path.with_file_name(ANE_SETUP_MARKER), ANE_SETUP_LIMIT, ane, on_cpu).map_err(for_ane)?;
    let want = want.map_err(|e| format!("the CPU could not check the Neural Engine: {e}"))?;
    let scale = want.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-12);
    let worst = want.iter().zip(&got).fold(0f32, |m, (a, b)| m.max((a - b).abs())) / scale;
    if got.len() != want.len() || !got.iter().all(|v| v.is_finite()) || worst > ANE_AGREE {
        return Err(format!("its answer on the check tile is not the CPU's (off by {worst:.1e} of the largest value): not used"));
    }
    Ok(AneSide { runner, ane_ms, cpu_ms })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn half_made_models_older_than_an_hour_go_and_others_stay() {
        let root = std::env::temp_dir().join(format!("lc-ane-stale-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let dir = root.join("abcd-t64-v1");
        let (old, fresh, other, kept) =
            (root.join("abcd-t64-v1.part-9-1"), root.join("abcd-t64-v1.part-9-2"), root.join("zzzz-t64-v1.part-9-1"), dir.clone());
        for d in [&old, &fresh, &other, &kept] {
            std::fs::create_dir_all(d).unwrap();
        }
        let two_hours_ago = SystemTime::now() - Duration::from_secs(7200);
        for d in [&old, &other, &kept] {
            std::fs::File::open(d).unwrap().set_modified(two_hours_ago).unwrap();
        }
        remove_stale_parts(&dir);
        assert!(!old.exists(), "a stale half-made model goes");
        assert!(fresh.exists(), "one that may still be made stays");
        assert!(other.exists(), "another model's stays");
        assert!(kept.exists(), "a finished model is never touched");
        // nothing to look at, nothing to fail
        remove_stale_parts(Path::new("/nonexistent/x"));
        remove_stale_parts(Path::new("/"));
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn a_panic_says_what_it_said() {
        use crate::denoise::gpu::panic_text;
        let literal = std::panic::catch_unwind(|| panic!("boom")).unwrap_err();
        assert_eq!(panic_text(&*literal), "boom");
        let formatted = std::panic::catch_unwind(|| panic!("{} {}", "a", 2)).unwrap_err();
        assert_eq!(panic_text(&*formatted), "a 2");
        let other = std::panic::catch_unwind(|| std::panic::panic_any(5u8)).unwrap_err();
        assert_eq!(panic_text(&*other), "no message");
    }

    #[test]
    fn the_switches_that_turn_accelerators_off_are_read_as_the_gpu_reads_them() {
        for off in ["0", "off", "false", "no", " 0 "] {
            assert!(gpu_env_off(Some(off)), "{off}");
        }
        for on in ["1", "on", "", "auto"] {
            assert!(!gpu_env_off(Some(on)), "{on}");
        }
        assert!(!gpu_env_off(None));
    }
}
