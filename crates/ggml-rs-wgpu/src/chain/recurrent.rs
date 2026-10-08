//! The recorder's delta net of a prompt's rows.
use super::*;

impl Recorder<'_> {
    /// [`ChainRecorder::delta_net`] for a prompt's rows ([`DELTA_NET_PREP`], [`DELTA_NET_SCAN`], [`DELTA_NET_NORM`]):
    /// every token's q, k and gates at once, then the recurrence a thread a state's row, then every token's norm.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn delta_net_rows(&mut self, conv: &DeviceVec, z: &DeviceVec, beta_alpha: &DeviceVec, ssm_a: &DeviceVec, dt_bias: &DeviceVec, norm: &DeviceVec, state: &DeviceVec, out: &DeviceVec, d: &DeltaNet, words: &[u32]) {
        let names: [&'static str; 3] = match d.k_dim {
            32 => ["chain-delta-net-prep-32", "chain-delta-net-scan-32", "chain-delta-net-norm-32"],
            64 => ["chain-delta-net-prep-64", "chain-delta-net-scan-64", "chain-delta-net-norm-64"],
            128 => ["chain-delta-net-prep-128", "chain-delta-net-scan-128", "chain-delta-net-norm-128"],
            other => panic!("chain: a delta net of heads of {other}"),
        };
        let dk = d.k_dim;
        let qk = self.scratch(d.rows * d.k_heads * 2 * dk + d.rows * d.v_heads * 2);
        let dd = self.gpu().dummy().clone();
        let drw = self.gpu().dummy_rw().clone();
        // a workgroup a warp's 32 rows (more of them than of heads: 0.32 ms a layer of Qwen3.8 27B's for 512 tokens,
        // as a head's 128; 64 0.44)
        let r = 32;
        let size = |s: &str| s.replace("DK_VALUE", &dk.to_string());
        let rows = d.rows as u32;
        self.dispatch_wide(names[0], &size(DELTA_NET_PREP), [buffer(conv), &dd, buffer(beta_alpha), buffer(ssm_a), buffer(dt_bias), &dd, buffer(&qk), &drw], words, (d.k_heads as u32, rows, 1));
        let groups = (d.v_heads * dk / r) as u32;
        self.dispatch_wide(names[1], &delta_net_scan(dk, r), [buffer(&qk), buffer(conv), buffer(&qk), &dd, &dd, &dd, buffer(state), buffer(out)], words, (groups, 1, 1));
        self.dispatch_wide(names[2], &size(DELTA_NET_NORM), [&dd, buffer(z), &dd, &dd, &dd, buffer(norm), &drw, buffer(out)], words, (d.v_heads as u32, rows, 1));
        // (the scan was the scratch's last reader: the next layer's delta net takes it again. Each of Flash-Next's
        // 36 such layers held its own to the recording's end, 16 MB at 512 rows: half of what a chunk took)
        self.done_with(&qk);
    }
}
