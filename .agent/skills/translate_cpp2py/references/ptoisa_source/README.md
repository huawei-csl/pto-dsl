If only need one file, use:
- https://gitcode.com/cann/pto-isa/blob/8.5.0/include/pto/common/pto_instr.hpp

Full references:
- https://gitcode.com/cann/pto-isa/tree/8.5.0/include/pto/npu/a2a3
- https://gitcode.com/cann/pto-isa/tree/8.5.0/include/pto/common

Remove function body and only keep header part, using script:

```bash
cd /Users/dalantianshi/work_code/pto-dsl/.agent/skills/translate_cpp2py/

# Extract a2a3_full -> a2a3_header
python3 scripts/extract_isa_header.py \
  --src references/ptoisa_source/a2a3_full

# Extract common_full -> common_header
python3 scripts/extract_isa_header.py \
  --src references/ptoisa_source/common_full
```
