use std::cell::{Cell, RefCell};

use anyhow::Result;
use pegainfer_kernels::tensor::KernelCall;

thread_local! {
    static TRACE: RefCell<Option<Vec<KernelCall>>> = const { RefCell::new(None) };
    static LABEL_STACK: RefCell<Vec<String>> = const { RefCell::new(Vec::new()) };
    static DECODE_KV_LEN: Cell<Option<usize>> = const { Cell::new(None) };
}

pub fn collect_result<T>(f: impl FnOnce() -> Result<T>) -> Result<(T, Vec<KernelCall>)> {
    TRACE.with(|trace| {
        let previous = trace.replace(Some(Vec::new()));
        assert!(
            previous.is_none(),
            "nested kernel call trace collection is not supported"
        );
    });

    let result = f();
    let calls = TRACE.with(|trace| trace.replace(None).unwrap_or_default());
    result.map(|value| (value, calls))
}

pub fn is_enabled() -> bool {
    TRACE.with(|trace| trace.borrow().is_some())
}

pub fn record_call(call: KernelCall) {
    TRACE.with(|trace| {
        if let Some(calls) = trace.borrow_mut().as_mut() {
            calls.push(call);
        }
    });
}

pub fn with_label<T>(label: impl Into<String>, f: impl FnOnce() -> T) -> T {
    LABEL_STACK.with(|stack| stack.borrow_mut().push(label.into()));
    let result = f();
    LABEL_STACK.with(|stack| {
        stack.borrow_mut().pop();
    });
    result
}

pub fn current_label(default_op: &str) -> String {
    LABEL_STACK.with(|stack| {
        stack
            .borrow()
            .last()
            .cloned()
            .unwrap_or_else(|| default_op.to_string())
    })
}

pub fn with_decode_kv_len<T>(kv_len: usize, f: impl FnOnce() -> T) -> T {
    DECODE_KV_LEN.with(|cell| {
        let previous = cell.replace(Some(kv_len));
        let result = f();
        cell.set(previous);
        result
    })
}

pub fn decode_kv_len() -> Option<usize> {
    DECODE_KV_LEN.with(Cell::get)
}
