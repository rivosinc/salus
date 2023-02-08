OUTPUT_ARCH( "riscv" )

ENTRY( _start )

PHDRS
{
    text PT_LOAD;
    rodata PT_LOAD;
    data PT_LOAD;
    stack PT_LOAD;
}

SECTIONS
{
    . = 0xffffffff00000000;

    .text ALIGN(4096) : {
        *(.text.start)
        *(.text.init) *(.text .text.*)
    } :text

    .rodata ALIGN(4096) : {
        *(.rodata .rodata.*)
    } :rodata

    .data ALIGN(4096) : {
        *(.data .data.*)

        . = ALIGN(8);
        PROVIDE(__global_pointer$ = . + 0x800);
        *(.sdata .sdata.*)

        *(.sbss .sbss.*) *(.bss .bss.*)
    } :data

    . += 16*1024*1024;

    .stack ALIGN(4096) (NOLOAD) : {
    PROVIDE(_stack_start = .);
    . += 176*1024;
    PROVIDE(_stack_end = .);
    } :stack

    /DISCARD/ : {
        *(.eh_frame)
    }
}
