/* SPDX-FileCopyrightText: 2023 Rivos Inc.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

use std::fs::File;
use std::io::Seek;
use std::io::{copy, BufReader, BufWriter, SeekFrom};

use std::env;

fn main() {
    // Parse Arguments
    let mut arg_list = env::args().skip(1);

    let file1 = arg_list.next().expect("No Tellus path");
    let file2 = arg_list.next().expect("No Guestvm path");
    let file3 = arg_list.next().expect("No Output path");
    let offset = arg_list.next().expect("No Max Tellus Size");

    let offset: u64 = offset.parse().expect("Invalid Max Tellus Size");

    // open files, check length
    let f1 = File::open(file1).expect("error reading Tellus");
    let tellus_len = f1.metadata().expect("Error getting length of Tellus").len();
    if tellus_len > offset {
        panic!("Tellus longer than Max Tellus size!");
    }
    let f2 = File::open(file2).expect("error reading Guestvm");
    let mut f3 = File::create(file3).expect("error opening Output");

    // create reader/writer
    let mut buf_reader1 = BufReader::new(&f1);
    let mut buf_reader2 = BufReader::new(&f2);
    let mut buf_writer = BufWriter::new(&mut f3);

    // copy into new file
    let wrote = copy(&mut buf_reader1, &mut buf_writer).expect("error writing Tellus to Output");
    if wrote > offset {
        panic!("Tellus longer than Max Tellus size!");
    }
    buf_writer
        .seek(SeekFrom::Start(offset))
        .expect("Problem Seeking in Output");
    copy(&mut buf_reader2, &mut buf_writer).expect("error writing Guestvm to Output");
}
