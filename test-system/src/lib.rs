// Copyright (c) 2021 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#![no_std]

//! Crate for basic test system that works with the docker
//! image ```rivos/rivos-qemu-docker```
//!
//! First, declare the test using test_declare,
//! `test_declare!("test_name")`
//! `test_declare!("test_name", hartid)`
//!
//! For a pass or fail use test_pass or test fail
//! `test_pass!("test_name")`
//! `test_pass!("test_name", hartid)`
//! `test_fail!("test_name")`
//! `test_fail!("test_name", hartid)`
//!
//! Test assert will check the boolean expression and
//! then call test_pass or test_fail
//! `test_assert!(boolean_expr, "test_name")`
//! `test_assert!(boolean_expr, "test_name", hartid)`

/// Simple enum for the result of a test
pub type TestResult = Result<(), TestFailure>;

/// Simple enum for failure result of a test
pub enum TestFailure {
    /// The test returns incorrectly
    Fail,
    /// The test fails, plus some info about what failed
    FailedAt(&'static str),
}

/// Declare and pass a test, for tests that
/// pass by reach a point in the code
#[macro_export]
macro_rules! test_declare_pass {
    ($n:expr) => {
        test_declare!($n);
        test_pass!($n)
    };
    ($n:expr,$h:expr) => {
        test_declare!($n, $h);
        test_pass!($n, $h)
    };
}

/// Declare and fail a test, for tests that
/// fail by reach a point in the code
#[macro_export]
macro_rules! test_declare_fail {
    ($n:expr) => {
        test_declare!($n);
        test_fail!($n)
    };
    ($n:expr,$h:expr) => {
        test_declare!($n);
        test_fail!($n)
    };
}

/// Declare a test
#[macro_export]
macro_rules! test_declare {
    ($n:expr) => {
        println!("TEST: {}", $n)
    };
    ($n:expr,$h:expr) => {
        println!("TEST: {} {}", $n, $h)
    };
}

/// Mark a test as failing
#[macro_export]
macro_rules! test_pass {
    ($n:expr) => {
        println!("PASS: {}", $n)
    };
    ($n:expr, $h:expr) => {
        println!("PASS: {} {}", $n, $h)
    };
}

/// Mark a test as passing
#[macro_export]
macro_rules! test_fail {
    ($n:expr) => {
        println!("FAIL: {}", $n)
    };
    ($n:expr, $h:expr) => {
        println!("FAIL: {} {}", $n, $h)
    };
}

/// Mark a test as passing or failing depending on the value
/// a boolean expression.
#[macro_export]
macro_rules! test_assert {
    ($e:expr, $n:expr) => {
        test_declare!($n);
        if $e {
            test_pass!($n)
        } else {
            test_fail!($n)
        }
    };
    ($e:expr, $n:expr, $h:expr) => {
        test_declare!($n, $h);
        if $e {
            test_pass!($n, $h)
        } else {
            test_fail!($n, $h)
        }
    };
}

/// Run a test in a block of code.
/// The block returns TestResult::Pass
/// or TestResult::Fail
///
///```
/// use test_system::*;
///
/// test_runtest!("runtest test", {
///     let a = 7;
///     if a - 6 == 1 {
///         Ok(())
///     } else {
///         Err(TestFailure::Fail)
///     }
/// });
/// ```
#[macro_export]
macro_rules! test_runtest {
    ($n:expr, $b:block) => {
        test_declare!($n);
        let res: TestResult = $b;
        match res {
            Ok(()) => test_pass!($n),
            Err(TestFailure::Fail) => test_fail!($n),
            Err(TestFailure::FailedAt(s)) => test_fail!($n, s),
        }
    };
}

/// Go from boolean to a TestResult, expecting true
#[macro_export]
macro_rules! true_to_tr {
    ($r:expr) => {
        if $r {
            Ok(())
        } else {
            Err(TestFailure::Fail)
        }
    };
    ($r: expr, $n:expr) => {
        if $r {
            Ok(())
        } else {
            Err(TestFailure::FailedAt($n))
        }
    };
}

/// Go from boolean to a TestResult, expecting false
#[macro_export]
macro_rules! false_to_tr {
    ($r:expr) => {
        if !$r {
            Ok(())
        } else {
            Err(TestFailure::Fail)
        }
    };
    ($r:expr, $n:expr) => {
        if !$r {
            Ok(())
        } else {
            Err(TestFailure::FailedAt($n))
        }
    };
}

/// run a test expecting true
#[macro_export]
macro_rules! test_result_true {
    // No name
    ($r:expr) => {
        true_to_tr!($r)
    };
    // with name
    ($r:expr, $n: expr) => {{
        if $r {
            test_declare_pass!($n);
        } else {
            test_declare_fail!($n);
        }
        true_to_tr!($r, $n)
    }};
}

/// run a test expecting false
#[macro_export]
macro_rules! test_result_false {
    // No name
    ($r:expr) => {
        false_to_tr!($r)
    };
    // with name
    ($r:expr, $n: expr) => {{
        if !$r {
            test_declare_pass!($n);
        } else {
            test_declare_fail!($n);
        }
        false_to_tr!($r, $n)
    }};
}
