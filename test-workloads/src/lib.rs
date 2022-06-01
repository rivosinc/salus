#![no_std]

use s_mode_utils::abort::abort;
use s_mode_utils::print_sbi::*;

mod asm;

#[panic_handler]
pub fn panic(info: &core::panic::PanicInfo) -> ! {
    println!("panic : {:?}", info);
    abort();
}
