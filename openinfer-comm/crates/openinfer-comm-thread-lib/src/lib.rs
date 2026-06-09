use libc::{
    CPU_SET, CPU_ZERO, cpu_set_t, pthread_self, pthread_setaffinity_np, sched_getcpu,
};
use syscalls::Errno;

#[inline]
pub fn current_tid_and_cpu() -> (i64, i32) {
    unsafe {
        let tid = libc::syscall(libc::SYS_gettid) as i64;
        let cpu = sched_getcpu();
        (tid, cpu)
    }
}

pub fn pin_cpu(cpu: usize) -> Result<(), Errno> {
    unsafe {
        let (tid, cpu_before) = current_tid_and_cpu();
        let mut cpuset = std::mem::zeroed();
        CPU_ZERO(&mut cpuset);
        CPU_SET(cpu, &mut cpuset);
        let ret =
            pthread_setaffinity_np(pthread_self(), size_of::<cpu_set_t>(), &cpuset);
        if ret != 0 {
            return Err(Errno::new(ret));
        }
        let (_, cpu_after) = current_tid_and_cpu();
        tracing::debug!(
            "[pin_cpu] tid={} target_cpu={} sched_cpu_before={} sched_cpu_after={}",
            tid,
            cpu,
            cpu_before,
            cpu_after
        );
        Ok(())
    }
}
