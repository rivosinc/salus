// Copyright (c) 2021 The RustCrypto Project Developers
// Copyright (c) 2022 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

//! X.501 time types as defined in RFC 5280

use core::fmt;
use core::time::Duration;
use der::asn1::{GeneralizedTime, UtcTime};
use der::{Choice, DateTime, Decode, Error, Result, Sequence};

/// X.501 `Time` as defined in [RFC 5280 Section 4.1.2.5].
///
/// Schema definition from [RFC 5280 Appendix A]:
///
/// ```text
/// Time ::= CHOICE {
///      utcTime        UTCTime,
///      generalTime    GeneralizedTime
/// }
/// ```
///
/// [RFC 5280 Section 4.1.2.5]: https://tools.ietf.org/html/rfc5280#section-4.1.2.5
/// [RFC 5280 Appendix A]: https://tools.ietf.org/html/rfc5280#page-117
#[derive(Choice, Copy, Clone, Debug, Eq, PartialEq)]
pub enum Time {
    /// Legacy UTC time (has 2-digit year, valid only through 2050).
    #[asn1(type = "UTCTime")]
    UtcTime(UtcTime),

    /// Modern [`GeneralizedTime`] encoding with 4-digit year.
    #[asn1(type = "GeneralizedTime")]
    GeneralTime(GeneralizedTime),
}

impl Time {
    /// Get duration since `UNIX_EPOCH`.
    pub fn to_unix_duration(self) -> Duration {
        match self {
            Time::UtcTime(t) => t.to_unix_duration(),
            Time::GeneralTime(t) => t.to_unix_duration(),
        }
    }

    /// Get Time as DateTime
    pub fn to_date_time(self) -> DateTime {
        match self {
            Time::UtcTime(t) => t.to_date_time(),
            Time::GeneralTime(t) => t.to_date_time(),
        }
    }

    /// Time in the past (The TCG DICE publication date: `180322235959Z``)
    pub fn past() -> Result<Self> {
        Ok(Time::UtcTime(UtcTime::from_date_time(DateTime::new(
            2018, 3, 22, 23, 59, 59,
        )?)?))
    }

    /// No well-known expiry date: `991231235959Z`
    pub fn never() -> Result<Self> {
        Ok(Time::UtcTime(UtcTime::from_date_time(DateTime::new(
            2049, 12, 31, 23, 59, 59,
        )?)?))
    }
}

impl fmt::Display for Time {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> core::result::Result<(), fmt::Error> {
        write!(f, "{}", self.to_date_time())
    }
}

impl From<UtcTime> for Time {
    fn from(time: UtcTime) -> Time {
        Time::UtcTime(time)
    }
}

impl From<GeneralizedTime> for Time {
    fn from(time: GeneralizedTime) -> Time {
        Time::GeneralTime(time)
    }
}

/// X.501 `Validity` as defined in [RFC 5280 Section 4.1.2.5]
///
/// ```text
/// Validity ::= SEQUENCE {
///     notBefore      Time,
///     notAfter       Time
/// }
/// ```
/// [RFC 5280 Section 4.1.2.5]: https://datatracker.ietf.org/doc/html/rfc5280#section-4.1.2.5
#[derive(Copy, Clone, Debug, Eq, PartialEq, Sequence)]
pub struct Validity {
    /// notBefore value
    pub not_before: Time,

    /// notAfter value
    pub not_after: Time,
}

impl<'a> TryFrom<&'a [u8]> for Validity {
    type Error = Error;

    fn try_from(bytes: &'a [u8]) -> Result<Self> {
        Self::from_der(bytes)
    }
}
