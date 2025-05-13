(module
  (type (;0;) (func (param i32 i32 i32 i32) (result i32)))
  (func (;0;) (type 0) (param i32 i32 i32 i32) (result i32)
    (local i32)
    local.get 0
    local.get 1
    local.get 2
    local.get 3
    i32.mul
    local.get 3
    i32.add
    i32.sub
    local.get 2
    i32.mul
    i32.add
    local.get 0
    local.get 2
    i32.mul
    local.get 0
    i32.sub
    i32.add
    local.get 0
    i32.mul
    local.get 3
    i32.xor
    drop
    local.get 1
    i32.const 100711 ;; random number larger than 2^16
    i32.and
    local.get 0
    i32.or
    local.get 1
    i32.xor
    local.get 3
    i32.const 230521 ;; random number larger than 2^16
    i32.and
    i32.or
    local.get 0
    i32.shl
    local.get 1
    i32.shr_s
    local.get 2
    i32.shl
    local.get 3
    i32.shr_u
    i32.const 100711 ;; random number larger than 2^16
    i32.and
    local.get 1
    i32.shr_s
    local.get 2
    i32.shl
    local.get 3
    i32.lt_s
    local.get 0
    i32.const 100711 ;; random number larger than 2^16
    i32.and
    i32.lt_u
  )
  (table (;0;) 1 1 funcref)
  (memory (;0;) 16)
  (global (;0;) (mut i32) (i32.const 1048576))
  (global (;1;) i32 (i32.const 1048576))
  (global (;2;) i32 (i32.const 1048576))
  (export "memory" (memory 0))
  (export "main" (func 0))
  (export "__data_end" (global 1))
  (export "__heap_base" (global 2)))
