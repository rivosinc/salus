Crate that handles riscv page table setup.

# Design

## Components

1. `Page`s - 4k/2M/1G/512G pages that span all system memory.
2. `PageTable` - Handles the configuration of system page tables to grant access
   to restricted memory for guests. Also owns the pages that are assigned to
   that guest.
3. guests - OSes running in VS mode.
4. host - The primary guest running in VS mode.
5. hypervisor - The HS mode code that uses `riscv-page-tables` to configure memory.

## Pages 

### Page ownership

Each page is owned by the page table it is assigned to. This is achieved by moving
it in to the guest's PageTable object when assigning it.  Except for shared
pages(revisted below).

Usage:

Initially, all RAM except for a minimum needed by the the hypervisor will be
assigned to a primary guest. That guest can later choose to assign some pages
to guests that it starts. This is achieved by moving the `Page` from the
donating guest's page tables into the new guests's page tables.

### Hypervisor owned pages

Some amount of memory is needed for hypervisor code and data. Those pages will
be owned by the hypervisor.

- [ ] TODO - Should there be an sv48 page table configured for the hypervisor and it
can run with paging enabled in HS mode? Then maybe the hypervisor can have a
limited allocator.

### shared pages

Pages that are mapped to multiple guests.

Used for cross-guest communication and sharing data.

TODO - is the right thing to do just to have each guest keep a set of Arc<Page>
for each page it owns?  This will allow maximal flexibility for shared pages,
and allow pages to be totally tracked by these ref counts, but will cost
memory.

Requires a secure way for each end to acknowledge the shared pages.

- [ ] TODO - Do they need a refcount? Just stick them in an Arc?
- [ ] TODO - How to unshare? - sync logic should be in upper layers, how to handle faults? - can start by only unsharing when shutting down a VM.
