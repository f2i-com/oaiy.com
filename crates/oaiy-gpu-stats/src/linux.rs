//! Linux: the kernel's DRM devices (`/sys/class/drm/cardN/device`), as `amdgpu_top` and `nvtop` read them where there
//! is no vendor tool: AMD's driver says its load (`gpu_busy_percent`) and the bytes of its own memory in use
//! (`mem_info_vram_used`); every driver with sensors says its temperature through hwmon (`hwmon/hwmonN/tempK_input`,
//! in millidegrees): amdgpu's `edge`, Intel's xe `pkg`, nouveau's one. std alone: these are files.

use crate::Gpu;
use std::path::Path;

/// Every DRM card under `root` (`/sys/class/drm` on a computer; another folder laid out as it is, in a test), in the
/// order of their numbers, with what its driver says.
pub(crate) fn read(root: &Path) -> Vec<Gpu> {
    let Ok(dir) = std::fs::read_dir(root) else { return Vec::new() };
    let mut cards: Vec<(u32, std::path::PathBuf)> = dir
        .filter_map(|e| e.ok())
        .filter_map(|e| {
            let name = e.file_name().into_string().ok()?;
            // (cardN alone: cardN-DP-1 and the like are its connectors)
            let n = name.strip_prefix("card")?.parse::<u32>().ok()?;
            Some((n, e.path().join("device")))
        })
        .collect();
    cards.sort();
    cards.into_iter().filter_map(|(_, device)| card(&device)).collect()
}

fn text(path: &Path) -> Option<String> {
    std::fs::read_to_string(path).ok().map(|s| s.trim().to_string()).filter(|s| !s.is_empty())
}

fn hex(path: &Path) -> Option<u32> {
    u32::from_str_radix(text(path)?.trim_start_matches("0x"), 16).ok()
}

fn card(device: &Path) -> Option<Gpu> {
    let vendor_id = hex(&device.join("vendor"))?;
    Some(Gpu {
        vendor_id: Some(vendor_id),
        device_id: hex(&device.join("device")),
        load: text(&device.join("gpu_busy_percent")).and_then(|v| v.parse::<i64>().ok()).filter(|p| (0..=100).contains(p)),
        memory_in_use: text(&device.join("mem_info_vram_used")).and_then(|v| v.parse::<i64>().ok()).filter(|b| *b >= 0),
        temperature: temperature(device),
    })
}

/// The GPU's temperature from its hwmon sensors: the one labelled as the GPU's own (`edge`, `pkg`, `gpu`) where there
/// are labels, else the first.
fn temperature(device: &Path) -> Option<f64> {
    let Ok(dir) = std::fs::read_dir(device.join("hwmon")) else { return None };
    let mut sensors: Vec<(u32, Option<String>, f64)> = Vec::new();
    for hwmon in dir.filter_map(|e| e.ok()).map(|e| e.path()) {
        let Ok(files) = std::fs::read_dir(&hwmon) else { continue };
        for name in files.filter_map(|e| e.ok()).filter_map(|e| e.file_name().into_string().ok()) {
            let Some(k) = name.strip_prefix("temp").and_then(|n| n.strip_suffix("_input")).and_then(|n| n.parse::<u32>().ok()) else { continue };
            let Some(milli) = text(&hwmon.join(&name)).and_then(|v| v.parse::<f64>().ok()) else { continue };
            let label = text(&hwmon.join(format!("temp{k}_label"))).map(|l| l.to_ascii_lowercase());
            sensors.push((k, label, milli / 1000.0));
        }
    }
    sensors.retain(|(_, _, t)| (1.0..130.0).contains(t));
    sensors.sort_by_key(|(k, _, _)| *k);
    ["edge", "pkg", "gpu"]
        .iter()
        .find_map(|want| sensors.iter().find(|(_, l, _)| l.as_deref() == Some(*want)))
        .or(sensors.first())
        .map(|(_, _, t)| *t)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write(root: &Path, file: &str, text: &str) {
        let path = root.join(file);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, text).unwrap();
    }

    /// An AMD card (its load, memory and three sensors, `edge` the GPU's), an Intel Arc on xe (its package sensor, no
    /// load), nouveau's one sensor, a connector that is no card, and a card past card9, as sysfs lays them out.
    #[test]
    fn each_cards_driver_says_its_load_memory_and_temperature() {
        let root = std::env::temp_dir().join(format!("oaiy-gpu-stats-sysfs-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        write(&root, "card0/device/vendor", "0x1002\n");
        write(&root, "card0/device/device", "0x744c\n");
        write(&root, "card0/device/gpu_busy_percent", "37\n");
        write(&root, "card0/device/mem_info_vram_used", "4294967296\n");
        write(&root, "card0/device/hwmon/hwmon3/temp1_input", "51000\n");
        write(&root, "card0/device/hwmon/hwmon3/temp1_label", "edge\n");
        write(&root, "card0/device/hwmon/hwmon3/temp2_input", "63000\n");
        write(&root, "card0/device/hwmon/hwmon3/temp2_label", "junction\n");
        write(&root, "card0/device/hwmon/hwmon3/temp3_input", "70000\n");
        write(&root, "card0/device/hwmon/hwmon3/temp3_label", "mem\n");
        write(&root, "card0-DP-1/status", "connected\n");
        write(&root, "card1/device/vendor", "0x8086\n");
        write(&root, "card1/device/device", "0xe20b\n");
        write(&root, "card1/device/hwmon/hwmon5/temp2_input", "44000\n");
        write(&root, "card1/device/hwmon/hwmon5/temp2_label", "pkg\n");
        write(&root, "card1/device/hwmon/hwmon5/temp3_input", "48000\n");
        write(&root, "card1/device/hwmon/hwmon5/temp3_label", "vram\n");
        write(&root, "card10/device/vendor", "0x10de\n");
        write(&root, "card10/device/device", "0x2684\n");
        write(&root, "card10/device/hwmon/hwmon1/temp1_input", "38000\n");
        write(&root, "card2/device/vendor", "0x10de\n");
        write(&root, "card2/device/hwmon/hwmon2/temp1_input", "0\n");
        let gpus = read(&root);
        let _ = std::fs::remove_dir_all(&root);
        let got: Vec<_> = gpus.iter().map(|g| (g.vendor_id, g.device_id, g.load, g.memory_in_use, g.temperature)).collect();
        assert_eq!(
            got,
            vec![
                (Some(0x1002), Some(0x744c), Some(37), Some(4 << 30), Some(51.0)),
                (Some(0x8086), Some(0xe20b), None, None, Some(44.0)),
                // (a sensor at 0 is one not read: nothing said)
                (Some(0x10de), None, None, None, None),
                (Some(0x10de), Some(0x2684), None, None, Some(38.0)),
            ]
        );
        assert!(read(Path::new("/no/such/folder")).is_empty());
    }
}
