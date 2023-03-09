/* SPDX-FileCopyrightText: 2023 Rivos Inc.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

OUTPUT_ARCH( "riscv" )

ENTRY( _start )

MEMORY
{
    ram  (wxa) : ORIGIN = 0x80200000, LENGTH = 128M
}

PHDRS
{
    text PT_LOAD;
    data PT_LOAD;
    bss PT_LOAD;
}

SECTIONS
{
    .start : {
        *(.text.start)
    } >ram AT>ram :text

    .text : {
        *(.text.init) *(.text .text.*)
    } >ram AT>ram :text

    .rodata : {
        *(.rodata .rodata.*)
    } >ram AT>ram :text

    .extable : {
        . = ALIGN(8);
        PROVIDE(_extable_start = .);
        KEEP(*(.extable))
        PROVIDE(_extable_end = .);
    } >ram AT>ram :text

    .data : {
        . = ALIGN(4096);
        *(.data .data.*)

        . = ALIGN(8);
        PROVIDE(__global_pointer$ = . + 0x800);
        *(.sdata .sdata.*)
    } >ram AT>ram :data

    .bss : {
        . = ALIGN(8);
        PROVIDE(_bss_start = .);
        *(.sbss .sbss.*) *(.bss .bss.*)
        . = ALIGN(8);
        PROVIDE(_bss_end = .);
    } >ram AT>ram :bss

    PROVIDE(_stack_start = ALIGN(_bss_end, 4096));
    PROVIDE(_stack_end = _stack_start + 0x80000);

    /DISCARD/ : {
        *(.eh_frame)
    }
}
