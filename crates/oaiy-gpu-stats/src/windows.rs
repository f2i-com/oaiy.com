//! Windows: the graphics kernel's own statistics (D3DKMT, in gdi32), which Task Manager's GPU pages read, for every
//! vendor's driver alike (WDDM 2.4 and later, Windows 10 1803 on):
//!
//! - its load: each engine's running time (`D3DKMT_QUERYSTATISTICS_NODE`), the busiest engine's share of the time
//!   between two readings, as Task Manager's "GPU" is;
//! - the memory in use: the bytes resident in the adapter's own memory segments (`D3DKMT_QUERYSTATISTICS_SEGMENT`,
//!   those not an aperture onto the computer's), Task Manager's "Dedicated GPU memory";
//! - its temperature, where its driver reports one (`KMTQAITYPE_ADAPTERPERFDATA`, tenths of a degree): Task Manager's
//!   "GPU Temperature", which a discrete card's driver gives and an integrated one's mostly does not.
//!
//! Each adapter is named by its PCI vendor and device ids (`KMTQAITYPE_PHYSICALADAPTERDEVICEIDS`), as the display
//! class in the registry names it; Microsoft's software adapter is left out.

use crate::Gpu;
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};
use windows_sys::Wdk::Graphics::Direct3D::{
    D3DKMTCloseAdapter, D3DKMTEnumAdapters2, D3DKMTQueryAdapterInfo, D3DKMTQueryStatistics, D3DKMT_ADAPTERINFO, D3DKMT_ADAPTER_PERFDATA, D3DKMT_CLOSEADAPTER,
    D3DKMT_ENUMADAPTERS2, D3DKMT_QUERYADAPTERINFO, D3DKMT_QUERYSTATISTICS, D3DKMT_QUERYSTATISTICS_ADAPTER, D3DKMT_QUERYSTATISTICS_NODE, D3DKMT_QUERYSTATISTICS_QUERY_NODE,
    D3DKMT_QUERYSTATISTICS_QUERY_SEGMENT, D3DKMT_QUERYSTATISTICS_RESULT, D3DKMT_QUERYSTATISTICS_SEGMENT, D3DKMT_QUERYSTATISTICS_TYPE, D3DKMT_QUERY_DEVICE_IDS,
    KMTQAITYPE_ADAPTERPERFDATA, KMTQAITYPE_PHYSICALADAPTERDEVICEIDS, KMTQUERYADAPTERINFOTYPE,
};
use windows_sys::Win32::Foundation::LUID;

/// Microsoft's PCI vendor id: its Basic Render Driver, a software adapter.
const MICROSOFT: u32 = 0x1414;

/// Each engine's running time (100 ns units) at a reading, by adapter.
type Times = HashMap<(u32, i32), (Instant, Vec<i64>)>;

/// Every adapter the graphics kernel has, with what it said. An adapter's load needs a reading before it: at its first,
/// a second is taken 100 ms after.
pub(crate) fn read() -> Vec<Gpu> {
    static BEFORE: OnceLock<Mutex<Times>> = OnceLock::new();
    let mut before = BEFORE.get_or_init(|| Mutex::new(HashMap::new())).lock().unwrap_or_else(|p| p.into_inner());
    let mut out = Vec::new();
    for adapter in adapters() {
        let Some((vendor_id, device_id)) = ids(adapter.hAdapter) else { continue };
        if vendor_id == MICROSOFT {
            continue;
        }
        let luid = adapter.AdapterLuid;
        let key = (luid.LowPart, luid.HighPart);
        let (nodes, segments) = counts(luid).unwrap_or((0, 0));
        let mut times = engine_times(luid, nodes);
        let mut at = Instant::now();
        if !before.contains_key(&key) && times.is_some() {
            before.insert(key, (at, times.clone().unwrap_or_default()));
            std::thread::sleep(Duration::from_millis(100));
            times = engine_times(luid, nodes);
            at = Instant::now();
        }
        let load = times.as_ref().and_then(|now| {
            let (then, earlier) = before.get(&key)?;
            crate::busiest(earlier, now, at.duration_since(*then).as_secs_f64())
        });
        if let Some(now) = times {
            before.insert(key, (at, now));
        }
        out.push(Gpu { vendor_id: Some(vendor_id), device_id: Some(device_id), load, memory_in_use: dedicated(luid, segments), temperature: temperature(adapter.hAdapter) });
    }
    out
}

/// The adapters, each with a handle the graphics kernel opened for it (closed when it is dropped).
fn adapters() -> Vec<Opened> {
    let mut list = D3DKMT_ENUMADAPTERS2 { NumAdapters: 0, pAdapters: std::ptr::null_mut() };
    // SAFETY: with a null array the call writes only the count into `list`, a live local of the struct it takes.
    if unsafe { D3DKMTEnumAdapters2(&mut list) } != 0 || list.NumAdapters == 0 {
        return Vec::new();
    }
    let mut infos = vec![D3DKMT_ADAPTERINFO::default(); list.NumAdapters as usize];
    list.pAdapters = infos.as_mut_ptr();
    // SAFETY: `infos` has room for NumAdapters entries, as the struct says; the call writes at most that many, and
    // sets NumAdapters to how many it wrote (an adapter gone since the count: fewer).
    if unsafe { D3DKMTEnumAdapters2(&mut list) } != 0 {
        return Vec::new();
    }
    infos.truncate(list.NumAdapters as usize);
    infos.into_iter().map(Opened).collect()
}

/// An adapter's handle from D3DKMTEnumAdapters2, closed once when dropped.
struct Opened(D3DKMT_ADAPTERINFO);

impl std::ops::Deref for Opened {
    type Target = D3DKMT_ADAPTERINFO;
    fn deref(&self) -> &D3DKMT_ADAPTERINFO {
        &self.0
    }
}

impl Drop for Opened {
    fn drop(&mut self) {
        let close = D3DKMT_CLOSEADAPTER { hAdapter: self.0.hAdapter };
        // SAFETY: the handle is one D3DKMTEnumAdapters2 opened for this process, closed here once (Opened is not Clone).
        unsafe { D3DKMTCloseAdapter(&close) };
    }
}

/// One of the adapter's facts (`kind`) into `data`, the struct that kind is answered in: whether it was answered.
fn adapter_info<T>(adapter: u32, kind: KMTQUERYADAPTERINFOTYPE, data: &mut T) -> bool {
    let mut query = D3DKMT_QUERYADAPTERINFO { hAdapter: adapter, Type: kind, pPrivateDriverData: (data as *mut T).cast(), PrivateDriverDataSize: std::mem::size_of::<T>() as u32 };
    // SAFETY: `data` is a live, writable T, and the size passed is T's: the call writes at most that many bytes there.
    // The caller passes the struct `kind` is answered in (D3DKMT_QUERY_DEVICE_IDS, D3DKMT_ADAPTER_PERFDATA).
    unsafe { D3DKMTQueryAdapterInfo(&mut query) == 0 }
}

/// The PCI vendor and device ids of the adapter's first physical adapter.
fn ids(adapter: u32) -> Option<(u32, u32)> {
    let mut asked = D3DKMT_QUERY_DEVICE_IDS::default();
    adapter_info(adapter, KMTQAITYPE_PHYSICALADAPTERDEVICEIDS, &mut asked).then_some((asked.DeviceIds.VendorID, asked.DeviceIds.DeviceID))
}

/// °C, where the driver reports it (0: it does not).
fn temperature(adapter: u32) -> Option<f64> {
    let mut perf = D3DKMT_ADAPTER_PERFDATA::default();
    if !adapter_info(adapter, KMTQAITYPE_ADAPTERPERFDATA, &mut perf) || perf.Temperature == 0 {
        return None;
    }
    Some(perf.Temperature as f64 / 10.0).filter(|t| (1.0..130.0).contains(t))
}

/// One statistic of the adapter `luid`, for the node or segment `index` where `kind` is one of those.
fn statistics(luid: LUID, kind: D3DKMT_QUERYSTATISTICS_TYPE, index: u32) -> Option<D3DKMT_QUERYSTATISTICS_RESULT> {
    let mut query = D3DKMT_QUERYSTATISTICS { Type: kind, AdapterLuid: luid, ..Default::default() };
    if kind == D3DKMT_QUERYSTATISTICS_NODE {
        query.Anonymous.QueryNode = D3DKMT_QUERYSTATISTICS_QUERY_NODE { NodeId: index };
    } else if kind == D3DKMT_QUERYSTATISTICS_SEGMENT {
        query.Anonymous.QuerySegment = D3DKMT_QUERYSTATISTICS_QUERY_SEGMENT { SegmentId: index };
    }
    // SAFETY: `query` is a live struct of the type the call takes, zeroed but for its type, adapter and index (its
    // process handle null: the adapter's own statistics, not a process's); the call writes its QueryResult, so the
    // pointer is made from the mutable place (the binding declares it *const; one from `&query` may not be written).
    (unsafe { D3DKMTQueryStatistics(std::ptr::addr_of_mut!(query).cast_const()) } == 0).then_some(query.QueryResult)
}

/// The adapter's engines and memory segments.
fn counts(luid: LUID) -> Option<(u32, u32)> {
    let result = statistics(luid, D3DKMT_QUERYSTATISTICS_ADAPTER, 0)?;
    // SAFETY: an adapter's statistics are answered in the union's AdapterInformation; every bit pattern is a valid u32.
    let info = unsafe { result.AdapterInformation };
    Some((info.NodeCount.min(256), info.NbSegments.min(256)))
}

/// Each engine's running time so far (100 ns units).
fn engine_times(luid: LUID, nodes: u32) -> Option<Vec<i64>> {
    let times: Vec<i64> = (0..nodes)
        .filter_map(|node| statistics(luid, D3DKMT_QUERYSTATISTICS_NODE, node))
        // SAFETY: a node's statistics are answered in the union's NodeInformation; every bit pattern is a valid i64.
        .map(|result| unsafe { result.NodeInformation.GlobalInformation.RunningTime })
        .collect();
    (!times.is_empty() && times.len() == nodes as usize).then_some(times)
}

/// The bytes resident in the adapter's own memory segments (not an aperture onto the computer's memory); None where
/// it has none (an integrated GPU's memory is the computer's).
fn dedicated(luid: LUID, segments: u32) -> Option<i64> {
    let mut own = None;
    for segment in 0..segments {
        let Some(result) = statistics(luid, D3DKMT_QUERYSTATISTICS_SEGMENT, segment) else { continue };
        // SAFETY: a segment's statistics are answered in the union's SegmentInformation; its fields are plain integers.
        let info = unsafe { result.SegmentInformation };
        if info.Aperture == 0 {
            *own.get_or_insert(0i64) += info.BytesResident.min(i64::MAX as u64) as i64;
        }
    }
    own
}
