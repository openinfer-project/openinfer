use std::fs;

use libc::{CPU_SET, CPU_ZERO, cpu_set_t, pthread_self, pthread_setaffinity_np};
use syscalls::Errno;

/// Pin the current thread so the kernel scheduler will keep it on the
/// NUMA node containing `cpu`. We deliberately do **not** restrict the
/// thread to a single CPU: with multiple ranks in one process, single-CPU
/// pinning across N ranks creates cross-thread serialization on shared
/// CPUs and starves the scheduler of any flexibility for IRQ co-location,
/// kthread interference, or migration during bursty progress loops. A
/// NUMA-node-wide mask preserves locality (cache / IB NIC bus access)
/// while letting the kernel place the thread on any free sibling.
pub fn pin_cpu(cpu: usize) -> Result<(), Errno> {
    let cpus = numa_node_cpus_for(cpu).unwrap_or_else(|| vec![cpu]);
    unsafe {
        let mut cpuset: cpu_set_t = std::mem::zeroed();
        CPU_ZERO(&mut cpuset);
        for c in &cpus {
            CPU_SET(*c, &mut cpuset);
        }
        let ret =
            pthread_setaffinity_np(pthread_self(), size_of::<cpu_set_t>(), &cpuset);
        if ret != 0 {
            return Err(Errno::new(ret));
        }
        Ok(())
    }
}

/// Look up the CPU list of the NUMA node that owns `cpu` by reading
/// `/sys/devices/system/cpu/cpuN/node*/cpulist`. Returns `None` if sysfs
/// has nothing to tell us (e.g., non-NUMA kernel, restricted environment),
/// in which case the caller falls back to pinning to just `cpu`.
fn numa_node_cpus_for(cpu: usize) -> Option<Vec<usize>> {
    let cpu_dir = format!("/sys/devices/system/cpu/cpu{cpu}");
    let entries = fs::read_dir(&cpu_dir).ok()?;
    for entry in entries.flatten() {
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if let Some(rest) = name.strip_prefix("node")
            && rest.chars().all(|c| c.is_ascii_digit())
        {
            let cpulist = entry.path().join("cpulist");
            if let Ok(content) = fs::read_to_string(&cpulist) {
                return Some(parse_cpulist(content.trim()));
            }
        }
    }
    None
}

/// Parse the `0-23,48-71` cpulist format used by sysfs into a flat Vec.
fn parse_cpulist(s: &str) -> Vec<usize> {
    let mut out = Vec::new();
    for part in s.split(',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        if let Some((lo, hi)) = part.split_once('-') {
            if let (Ok(lo), Ok(hi)) = (lo.parse::<usize>(), hi.parse::<usize>()) {
                out.extend(lo..=hi);
            }
        } else if let Ok(v) = part.parse::<usize>() {
            out.push(v);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::parse_cpulist;

    #[test]
    fn parses_ranges_and_singletons() {
        assert_eq!(parse_cpulist("0-3,7,10-11"), vec![0, 1, 2, 3, 7, 10, 11]);
        assert_eq!(parse_cpulist(""), Vec::<usize>::new());
        assert_eq!(parse_cpulist("42"), vec![42]);
    }
}
