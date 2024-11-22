# MiniTorch Module 3

<img src="https://minitorch.github.io/minitorch.svg" width="50%">

* Docs: https://minitorch.github.io/

* Overview: https://minitorch.github.io/module3.html


You will need to modify `tensor_functions.py` slightly in this assignment.

* Tests:

```
python run_tests.py
```

* Note:

Several of the tests for this assignment will only run if you are on a GPU machine and will not
run on github's test infrastructure. Please follow the instructions to setup up a colab machine
to run these tests.

This assignment requires the following files from the previous assignments. You can get these by running

```bash
python sync_previous_module.py previous-module-dir current-module-dir
```

The files that will be synced are:

        minitorch/tensor_data.py minitorch/tensor_functions.py minitorch/tensor_ops.py minitorch/operators.py minitorch/scalar.py minitorch/scalar_functions.py minitorch/module.py minitorch/autodiff.py minitorch/module.py project/run_manual.py project/run_scalar.py project/run_tensor.py minitorch/operators.py minitorch/module.py minitorch/autodiff.py minitorch/tensor.py minitorch/datasets.py minitorch/testing.py minitorch/optim.py
## Task 3.1

```bash
MAP
OMP: Info #276: omp_set_nested routine deprecated, please use omp_set_max_active_levels instead.
 
================================================================================
 Parallel Accelerator Optimizing:  Function tensor_map.<locals>._map, 
/Users/reitwiec/Desktop/Cornell/MLE/mod3-reitwiec/minitorch/fast_ops.py (163)  
================================================================================


Parallel loop listing for  Function tensor_map.<locals>._map, /Users/reitwiec/Desktop/Cornell/MLE/mod3-reitwiec/minitorch/fast_ops.py (163) 
-----------------------------------------------------------------------------------------------------------------------------------------|loop #ID
    def _map(                                                                                                                            | 
        out: Storage,                                                                                                                    | 
        out_shape: Shape,                                                                                                                | 
        out_strides: Strides,                                                                                                            | 
        in_storage: Storage,                                                                                                             | 
        in_shape: Shape,                                                                                                                 | 
        in_strides: Strides,                                                                                                             | 
    ) -> None:                                                                                                                           | 
        # TODO: Implement for Task 3.1.                                                                                                  | 
        direct_mapping = len(out_strides) != len(in_strides) or (out_strides != in_strides).any() or (out_shape != in_shape).any()-------| #0, 1
        if direct_mapping:                                                                                                               | 
            for i in prange(len(out)):---------------------------------------------------------------------------------------------------| #3
                out_idx = np.empty(MAX_DIMS, np.int32)                                                                                   | 
                in_idx = np.empty(MAX_DIMS, np.int32)                                                                                    | 
                # compute multidimensional indices                                                                                       | 
                to_index(i, out_shape, out_idx)                                                                                          | 
                broadcast_index(out_idx, out_shape, in_shape, in_idx)                                                                    | 
                out_posn = index_to_position(out_idx, out_strides)                                                                       | 
                in_posn = index_to_position(in_idx, in_strides)                                                                          | 
                out[out_posn] = fn(in_storage[in_posn])                                                                                  | 
            return                                                                                                                       | 
        for i in prange(len(out)):-------------------------------------------------------------------------------------------------------| #2
            out[i] = fn(in_storage[i])                                                                                                   | 
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 4 parallel for-
loop(s) (originating from loops labelled: #0, #1, #3, #2).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
 
---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at 
/Users/reitwiec/Desktop/Cornell/MLE/mod3-reitwiec/minitorch/fast_ops.py (175) is
 hoisted out of the parallel loop labelled #3 (it will be performed before the 
loop is executed and reused inside the loop):
   Allocation:: out_idx = np.empty(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at 
/Users/reitwiec/Desktop/Cornell/MLE/mod3-reitwiec/minitorch/fast_ops.py (176) is
 hoisted out of the parallel loop labelled #3 (it will be performed before the 
loop is executed and reused inside the loop):
   Allocation:: in_idx = np.empty(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
None
ZIP
 
================================================================================
 Parallel Accelerator Optimizing:  Function tensor_zip.<locals>._zip, 
/Users/reitwiec/Desktop/Cornell/MLE/mod3-reitwiec/minitorch/fast_ops.py (213)  
================================================================================


Parallel loop listing for  Function tensor_zip.<locals>._zip, /Users/reitwiec/Desktop/Cornell/MLE/mod3-reitwiec/minitorch/fast_ops.py (213) 
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|loop #ID
    def _zip(                                                                                                                                                                                                                                           | 
        out: Storage,                                                                                                                                                                                                                                   | 
        out_shape: Shape,                                                                                                                                                                                                                               | 
        out_strides: Strides,                                                                                                                                                                                                                           | 
        a_storage: Storage,                                                                                                                                                                                                                             | 
        a_shape: Shape,                                                                                                                                                                                                                                 | 
        a_strides: Strides,                                                                                                                                                                                                                             | 
        b_storage: Storage,                                                                                                                                                                                                                             | 
        b_shape: Shape,                                                                                                                                                                                                                                 | 
        b_strides: Strides,                                                                                                                                                                                                                             | 
    ) -> None:                                                                                                                                                                                                                                          | 
        # TODO: Implement for Task 3.1.                                                                                                                                                                                                                 | 
        direct_mapping = (len(out_strides) != len(a_strides) or len(out_strides) != len(b_strides) or (out_strides != a_strides).any() or (out_strides != b_strides).any() or (out_shape != a_shape).any() or (out_shape != b_shape).any())-------------| #4, 5, 6, 7
        if direct_mapping:                                                                                                                                                                                                                              | 
            for i in prange(len(out)):------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------| #9
                out_idx = np.empty(MAX_DIMS, np.int32)                                                                                                                                                                                                  | 
                a_idx = np.empty(MAX_DIMS, np.int32)                                                                                                                                                                                                    | 
                b_idx = np.empty(MAX_DIMS, np.int32)                                                                                                                                                                                                    | 
                to_index(i, out_shape, out_idx)                                                                                                                                                                                                         | 
                broadcast_index(out_idx, out_shape, a_shape, a_idx)                                                                                                                                                                                     | 
                broadcast_index(out_idx, out_shape, b_shape, b_idx)                                                                                                                                                                                     | 
                a_posn = index_to_position(a_idx, a_strides)                                                                                                                                                                                            | 
                b_posn = index_to_position(b_idx, b_strides)                                                                                                                                                                                            | 
                a_data = a_storage[a_posn]                                                                                                                                                                                                              | 
                b_data = b_storage[b_posn]                                                                                                                                                                                                              | 
                out_posn = index_to_position(out_idx, out_strides)                                                                                                                                                                                      | 
                out[out_posn] = fn(a_data, b_data)                                                                                                                                                                                                      | 
            return                                                                                                                                                                                                                                      | 
        for i in prange(len(out)):----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------| #8
            out[i] = fn(a_storage[i], b_storage[i])                                                                                                                                                                                                     | 
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 6 parallel for-
loop(s) (originating from loops labelled: #4, #5, #6, #7, #9, #8).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
 
---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at 
/Users/reitwiec/Desktop/Cornell/MLE/mod3-reitwiec/minitorch/fast_ops.py (228) is
 hoisted out of the parallel loop labelled #9 (it will be performed before the 
loop is executed and reused inside the loop):
   Allocation:: out_idx = np.empty(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at 
/Users/reitwiec/Desktop/Cornell/MLE/mod3-reitwiec/minitorch/fast_ops.py (229) is
 hoisted out of the parallel loop labelled #9 (it will be performed before the 
loop is executed and reused inside the loop):
   Allocation:: a_idx = np.empty(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at 
/Users/reitwiec/Desktop/Cornell/MLE/mod3-reitwiec/minitorch/fast_ops.py (230) is
 hoisted out of the parallel loop labelled #9 (it will be performed before the 
loop is executed and reused inside the loop):
   Allocation:: b_idx = np.empty(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
None
REDUCE
 
================================================================================
 Parallel Accelerator Optimizing:  Function tensor_reduce.<locals>._reduce, 
/Users/reitwiec/Desktop/Cornell/MLE/mod3-reitwiec/minitorch/fast_ops.py (268)  
================================================================================


Parallel loop listing for  Function tensor_reduce.<locals>._reduce, /Users/reitwiec/Desktop/Cornell/MLE/mod3-reitwiec/minitorch/fast_ops.py (268) 
--------------------------------------------------------------------|loop #ID
    def _reduce(                                                    | 
        out: Storage,                                               | 
        out_shape: Shape,                                           | 
        out_strides: Strides,                                       | 
        a_storage: Storage,                                         | 
        a_shape: Shape,                                             | 
        a_strides: Strides,                                         | 
        reduce_dim: int,                                            | 
    ) -> None:                                                      | 
        # TODO: Implement for Task 3.1.                             | 
        for i in prange(len(out)):----------------------------------| #10
            out_index = np.empty(MAX_DIMS, np.int32)                | 
            dim = a_shape[reduce_dim]                               | 
            to_index(i, out_shape, out_index)                       | 
            out_posn = index_to_position(out_index, out_strides)    | 
            accum = out[out_posn]                                   | 
            posn = index_to_position(out_index, a_strides)          | 
            ast = a_strides[reduce_dim]                             | 
            for step in range(dim):                                 | 
                accum = fn(accum, a_storage[posn])                  | 
                posn += ast                                         | 
            out[out_posn] = accum                                   | 
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 1 parallel for-
loop(s) (originating from loops labelled: #10).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
 
---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at 
/Users/reitwiec/Desktop/Cornell/MLE/mod3-reitwiec/minitorch/fast_ops.py (279) is
 hoisted out of the parallel loop labelled #10 (it will be performed before the 
loop is executed and reused inside the loop):
   Allocation:: out_index = np.empty(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
None
MATRIX MULTIPLY
 
================================================================================
 Parallel Accelerator Optimizing:  Function _tensor_matrix_multiply, 
/Users/reitwiec/Desktop/Cornell/MLE/mod3-reitwiec/minitorch/fast_ops.py (294)  
================================================================================


Parallel loop listing for  Function _tensor_matrix_multiply, /Users/reitwiec/Desktop/Cornell/MLE/mod3-reitwiec/minitorch/fast_ops.py (294) 
----------------------------------------------------------------------------------------------------------|loop #ID
def _tensor_matrix_multiply(                                                                              | 
    out: Storage,                                                                                         | 
    out_shape: Shape,                                                                                     | 
    out_strides: Strides,                                                                                 | 
    a_storage: Storage,                                                                                   | 
    a_shape: Shape,                                                                                       | 
    a_strides: Strides,                                                                                   | 
    b_storage: Storage,                                                                                   | 
    b_shape: Shape,                                                                                       | 
    b_strides: Strides,                                                                                   | 
) -> None:                                                                                                | 
    """NUMBA tensor matrix multiply function.                                                             | 
                                                                                                          | 
    Should work for any tensor shapes that broadcast as long as                                           | 
                                                                                                          | 
    ```                                                                                                   | 
    assert a_shape[-1] == b_shape[-2]                                                                     | 
    ```                                                                                                   | 
                                                                                                          | 
    Optimizations:                                                                                        | 
                                                                                                          | 
    * Outer loop in parallel                                                                              | 
    * No index buffers or function calls                                                                  | 
    * Inner loop should have no global writes, 1 multiply.                                                | 
                                                                                                          | 
                                                                                                          | 
    Args:                                                                                                 | 
    ----                                                                                                  | 
        out (Storage): storage for `out` tensor                                                           | 
        out_shape (Shape): shape for `out` tensor                                                         | 
        out_strides (Strides): strides for `out` tensor                                                   | 
        a_storage (Storage): storage for `a` tensor                                                       | 
        a_shape (Shape): shape for `a` tensor                                                             | 
        a_strides (Strides): strides for `a` tensor                                                       | 
        b_storage (Storage): storage for `b` tensor                                                       | 
        b_shape (Shape): shape for `b` tensor                                                             | 
        b_strides (Strides): strides for `b` tensor                                                       | 
                                                                                                          | 
    Returns:                                                                                              | 
    -------                                                                                               | 
        None : Fills in `out`                                                                             | 
                                                                                                          | 
    """                                                                                                   | 
    # strides for broadcasting in batch dimensions.                                                       | 
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0                                                | 
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0                                                | 
                                                                                                          | 
    # TODO: Implement for Task 3.2.                                                                       | 
    # outer loop over the batch dimension of the output.                                                  | 
    for batch_idx in prange(out_shape[0]):----------------------------------------------------------------| #13
        for row_idx in prange(out_shape[1]):--------------------------------------------------------------| #12
            for col_idx in prange(out_shape[2]):----------------------------------------------------------| #11
                # calculate initial positions in A and B for the current batch and row/column indices.    | 
                a_pos = batch_idx * a_batch_stride + row_idx * a_strides[1]                               | 
                b_pos = batch_idx * b_batch_stride + col_idx * b_strides[2]                               | 
                                                                                                          | 
                # compute the dot product for the current row and column.                                 | 
                accumulator = 0.0                                                                         | 
                for k in range(a_shape[2]):  # loop over the common dimension.                            | 
                    accumulator += a_storage[a_pos] * b_storage[b_pos]                                    | 
                    a_pos += a_strides[2]                                                                 | 
                    b_pos += b_strides[1]                                                                 | 
                                                                                                          | 
                output_pos = (                                                                            | 
                    batch_idx * out_strides[0]                                                            | 
                    + row_idx * out_strides[1]                                                            | 
                    + col_idx * out_strides[2]                                                            | 
                )                                                                                         | 
                out[output_pos] = accumulator                                                             | 
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 2 parallel for-
loop(s) (originating from loops labelled: #13, #12).
--------------------------------------------------------------------------------
---------------------------- Optimising loop nests -----------------------------
Attempting loop nest rewrites (optimising for the largest parallel loops)...
 
+--13 is a parallel loop
   +--12 --> rewritten as a serial loop
      +--11 --> rewritten as a serial loop
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
Parallel region 0:
+--13 (parallel)
   +--12 (parallel)
      +--11 (parallel)


--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel region 0:
+--13 (parallel)
   +--12 (serial)
      +--11 (serial)


 
Parallel region 0 (loop #13) had 0 loop(s) fused and 2 loop(s) serialized as 
part of the larger parallel loop (#13).
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
 
---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
No allocation hoisting found
None
```


## Task 3.4

![Fast vs GPU](./assets/fastvgpu.png)
![Fast vs GPU Plot](./assets/fastvgpuplot.png)

## Task 3.5

## Split Dataset

python3.12 project/run_fast_tensor.py --BACKEND gpu --HIDDEN 100 --DATASET split --RATE 0.05

*Average time per epoch 1.3685702180862427 (for 500 epochs)*

<details>
  <summary>View log</summary>
```
Epoch  0  loss  6.979193031077563 correct 35
Epoch  1  time  1.5539555549621582
Epoch  2  time  1.7857658863067627
Epoch  3  time  1.2591421604156494
Epoch  4  time  1.3024561405181885
Epoch  5  time  1.2663373947143555
Epoch  6  time  1.2593190670013428
Epoch  7  time  1.3193223476409912
Epoch  8  time  1.2872166633605957
Epoch  9  time  1.4752295017242432
Epoch  10  time  1.6147856712341309
Epoch  10  loss  5.876658808095959 correct 42
Epoch  11  time  1.672334909439087
Epoch  12  time  1.2987086772918701
Epoch  13  time  1.2636969089508057
Epoch  14  time  1.2449183464050293
Epoch  15  time  1.2554285526275635
Epoch  16  time  1.247899055480957
Epoch  17  time  1.3093657493591309
Epoch  18  time  1.2518393993377686
Epoch  19  time  1.5572760105133057
Epoch  20  time  1.7416749000549316
Epoch  20  loss  4.94805701854571 correct 44
Epoch  21  time  1.3050627708435059
Epoch  22  time  1.392138957977295
Epoch  23  time  1.2564046382904053
Epoch  24  time  1.2480640411376953
Epoch  25  time  1.2569220066070557
Epoch  26  time  1.27284836769104
Epoch  27  time  1.3138811588287354
Epoch  28  time  1.598071575164795
Epoch  29  time  1.7332391738891602
Epoch  30  time  1.2830450534820557
Epoch  30  loss  4.542745641917164 correct 44
Epoch  31  time  1.2631471157073975
Epoch  32  time  1.316455602645874
Epoch  33  time  1.2647087574005127
Epoch  34  time  1.2688257694244385
Epoch  35  time  1.260362148284912
Epoch  36  time  1.2554638385772705
Epoch  37  time  1.6337854862213135
Epoch  38  time  1.7241153717041016
Epoch  39  time  1.2492485046386719
Epoch  40  time  1.2759017944335938
Epoch  40  loss  2.638147542803417 correct 41
Epoch  41  time  1.2746431827545166
Epoch  42  time  1.3226232528686523
Epoch  43  time  1.2478365898132324
Epoch  44  time  1.262939691543579
Epoch  45  time  1.273878574371338
Epoch  46  time  1.532459020614624
Epoch  47  time  1.8023216724395752
Epoch  48  time  1.2638826370239258
Epoch  49  time  1.257580041885376
Epoch  50  time  1.2626285552978516
Epoch  50  loss  2.868000854493328 correct 47
Epoch  51  time  1.2577857971191406
Epoch  52  time  1.3039450645446777
Epoch  53  time  1.2777104377746582
Epoch  54  time  1.256572961807251
Epoch  55  time  1.518470048904419
Epoch  56  time  1.8072142601013184
Epoch  57  time  1.3045616149902344
Epoch  58  time  1.258470058441162
Epoch  59  time  1.253685712814331
Epoch  60  time  1.2614598274230957
Epoch  60  loss  2.9011538892774316 correct 47
Epoch  61  time  1.24277663230896
Epoch  62  time  1.3013954162597656
Epoch  63  time  1.256472110748291
Epoch  64  time  1.465540885925293
Epoch  65  time  1.8423748016357422
Epoch  66  time  1.258965015411377
Epoch  67  time  1.2640810012817383
Epoch  68  time  1.3115427494049072
Epoch  69  time  1.2576713562011719
Epoch  70  time  1.2450220584869385
Epoch  70  loss  3.2326193544058537 correct 47
Epoch  71  time  1.2478601932525635
Epoch  72  time  1.2485876083374023
Epoch  73  time  1.4880728721618652
Epoch  74  time  1.8680837154388428
Epoch  75  time  1.2817113399505615
Epoch  76  time  1.2532505989074707
Epoch  77  time  1.274193525314331
Epoch  78  time  1.3224663734436035
Epoch  79  time  1.2644689083099365
Epoch  80  time  1.2633490562438965
Epoch  80  loss  2.5907709315687146 correct 48
Epoch  81  time  1.2514293193817139
Epoch  82  time  1.3855860233306885
Epoch  83  time  1.9541065692901611
Epoch  84  time  1.2504279613494873
Epoch  85  time  1.2722344398498535
Epoch  86  time  1.2535960674285889
Epoch  87  time  1.2576947212219238
Epoch  88  time  1.3358640670776367
Epoch  89  time  1.263749361038208
Epoch  90  time  1.2616522312164307
Epoch  90  loss  0.9603339505927388 correct 45
Epoch  91  time  1.3871185779571533
Epoch  92  time  1.8561842441558838
Epoch  93  time  1.4063324928283691
Epoch  94  time  1.2628448009490967
Epoch  95  time  1.2691631317138672
Epoch  96  time  1.2481212615966797
Epoch  97  time  1.2818026542663574
Epoch  98  time  1.3011162281036377
Epoch  99  time  1.2491633892059326
Epoch  100  time  1.3247814178466797
Epoch  100  loss  1.1615325489623687 correct 47
Epoch  101  time  1.853245735168457
Epoch  102  time  1.3654799461364746
Epoch  103  time  1.3196353912353516
Epoch  104  time  1.2601807117462158
Epoch  105  time  1.253232717514038
Epoch  106  time  1.2702927589416504
Epoch  107  time  1.2530920505523682
Epoch  108  time  1.315704107284546
Epoch  109  time  1.2496378421783447
Epoch  110  time  1.8423311710357666
Epoch  110  loss  2.267804223077592 correct 45
Epoch  111  time  1.4388954639434814
Epoch  112  time  1.2648608684539795
Epoch  113  time  1.3095974922180176
Epoch  114  time  1.2665815353393555
Epoch  115  time  1.2726798057556152
Epoch  116  time  1.2741444110870361
Epoch  117  time  1.2653014659881592
Epoch  118  time  1.302009105682373
Epoch  119  time  1.7970876693725586
Epoch  120  time  1.5126161575317383
Epoch  120  loss  1.7534978818670803 correct 49
Epoch  121  time  1.283637523651123
Epoch  122  time  1.2739875316619873
Epoch  123  time  1.563509225845337
Epoch  124  time  1.849381685256958
Epoch  125  time  1.2486391067504883
Epoch  126  time  1.272792100906372
Epoch  127  time  1.517918348312378
Epoch  128  time  1.7699592113494873
Epoch  129  time  1.3159520626068115
Epoch  130  time  1.2420098781585693
Epoch  130  loss  2.153089651454067 correct 48
Epoch  131  time  1.259817361831665
Epoch  132  time  1.2629752159118652
Epoch  133  time  1.2453186511993408
Epoch  134  time  1.3082044124603271
Epoch  135  time  1.2598369121551514
Epoch  136  time  1.5073227882385254
Epoch  137  time  1.8349480628967285
Epoch  138  time  1.25321364402771
Epoch  139  time  1.30999755859375
Epoch  140  time  1.2806577682495117
Epoch  140  loss  3.036527901001681 correct 44
Epoch  141  time  1.2434868812561035
Epoch  142  time  1.2522578239440918
Epoch  143  time  1.2745611667633057
Epoch  144  time  1.327730417251587
Epoch  145  time  1.4790301322937012
Epoch  146  time  1.8396739959716797
Epoch  147  time  1.2618205547332764
Epoch  148  time  1.2649109363555908
Epoch  149  time  1.315934419631958
Epoch  150  time  1.2679321765899658
Epoch  150  loss  1.311822059833642 correct 48
Epoch  151  time  1.2530584335327148
Epoch  152  time  1.2561454772949219
Epoch  153  time  1.263746738433838
Epoch  154  time  1.5307908058166504
Epoch  155  time  1.861574411392212
Epoch  156  time  1.2593226432800293
Epoch  157  time  1.2523033618927002
Epoch  158  time  1.2501347064971924
Epoch  159  time  1.3172709941864014
Epoch  160  time  1.2717034816741943
Epoch  160  loss  0.9753811673729593 correct 50
Epoch  161  time  1.2702436447143555
Epoch  162  time  1.2835116386413574
Epoch  163  time  1.3979768753051758
Epoch  164  time  1.9685063362121582
Epoch  165  time  1.255664348602295
Epoch  166  time  1.254302740097046
Epoch  167  time  1.2616944313049316
Epoch  168  time  1.2757411003112793
Epoch  169  time  1.3585224151611328
Epoch  170  time  1.2567377090454102
Epoch  170  loss  1.7920555057574692 correct 48
Epoch  171  time  1.2717325687408447
Epoch  172  time  1.413414716720581
Epoch  173  time  1.891056776046753
Epoch  174  time  1.3323051929473877
Epoch  175  time  1.2615907192230225
Epoch  176  time  1.2653064727783203
Epoch  177  time  1.255645513534546
Epoch  178  time  1.2546570301055908
Epoch  179  time  1.3118205070495605
Epoch  180  time  1.2599682807922363
Epoch  180  loss  1.1800014359570987 correct 50
Epoch  181  time  1.3876488208770752
Epoch  182  time  1.8633198738098145
Epoch  183  time  1.3461930751800537
Epoch  184  time  1.3413336277008057
Epoch  185  time  1.2454895973205566
Epoch  186  time  1.2508647441864014
Epoch  187  time  1.2672984600067139
Epoch  188  time  1.2499394416809082
Epoch  189  time  1.3026807308197021
Epoch  190  time  1.3022856712341309
Epoch  190  loss  0.22250212608747122 correct 50
Epoch  191  time  1.8489909172058105
Epoch  192  time  1.3836240768432617
Epoch  193  time  1.2609715461730957
Epoch  194  time  1.311880350112915
Epoch  195  time  1.2628235816955566
Epoch  196  time  1.2589902877807617
Epoch  197  time  1.2635619640350342
Epoch  198  time  1.2426671981811523
Epoch  199  time  1.3143727779388428
Epoch  200  time  1.8697261810302734
Epoch  200  loss  0.7012605437600207 correct 47
Epoch  201  time  1.4053192138671875
Epoch  202  time  1.2559187412261963
Epoch  203  time  1.244452714920044
Epoch  204  time  1.3213417530059814
Epoch  205  time  1.273655891418457
Epoch  206  time  1.2957446575164795
Epoch  207  time  1.2500154972076416
Epoch  208  time  1.2511420249938965
Epoch  209  time  1.8850934505462646
Epoch  210  time  1.4765548706054688
Epoch  210  loss  1.0891666748400002 correct 49
Epoch  211  time  1.248777151107788
Epoch  212  time  1.2511076927185059
Epoch  213  time  1.2626070976257324
Epoch  214  time  1.3025233745574951
Epoch  215  time  1.2810561656951904
Epoch  216  time  1.2596712112426758
Epoch  217  time  1.2701458930969238
Epoch  218  time  1.7854464054107666
Epoch  219  time  1.635930061340332
Epoch  220  time  1.2933449745178223
Epoch  220  loss  1.1057104247073923 correct 50
Epoch  221  time  1.2668042182922363
Epoch  222  time  1.24290132522583
Epoch  223  time  1.2585885524749756
Epoch  224  time  1.318199872970581
Epoch  225  time  1.2633838653564453
Epoch  226  time  1.2593867778778076
Epoch  227  time  1.773608922958374
Epoch  228  time  1.611375093460083
Epoch  229  time  1.3274590969085693
Epoch  230  time  1.268251895904541
Epoch  230  loss  0.8404650463473171 correct 47
Epoch  231  time  1.2471213340759277
Epoch  232  time  1.2505967617034912
Epoch  233  time  1.2618260383605957
Epoch  234  time  1.3038089275360107
Epoch  235  time  1.2485201358795166
Epoch  236  time  1.7314918041229248
Epoch  237  time  1.595989465713501
Epoch  238  time  1.2538316249847412
Epoch  239  time  1.307856798171997
Epoch  240  time  1.24674654006958
Epoch  240  loss  1.0555106808839836 correct 48
Epoch  241  time  1.268728494644165
Epoch  242  time  1.2824416160583496
Epoch  243  time  1.2674379348754883
Epoch  244  time  1.3166439533233643
Epoch  245  time  1.692880392074585
Epoch  246  time  1.6271774768829346
Epoch  247  time  1.2765278816223145
Epoch  248  time  1.2604472637176514
Epoch  249  time  1.312910556793213
Epoch  250  time  1.2886815071105957
Epoch  250  loss  1.6931930297703506 correct 47
Epoch  251  time  1.2622134685516357
Epoch  252  time  1.25474214553833
Epoch  253  time  1.2548589706420898
Epoch  254  time  1.7348895072937012
Epoch  255  time  1.6220018863677979
Epoch  256  time  1.2551243305206299
Epoch  257  time  1.267195463180542
Epoch  258  time  1.262984037399292
Epoch  259  time  1.3088946342468262
Epoch  260  time  1.254580020904541
Epoch  260  loss  1.0702682545775355 correct 49
Epoch  261  time  1.2565827369689941
Epoch  262  time  1.2541537284851074
Epoch  263  time  1.6147894859313965
Epoch  264  time  1.7747094631195068
Epoch  265  time  1.2538444995880127
Epoch  266  time  1.259751558303833
Epoch  267  time  1.2607240676879883
Epoch  268  time  1.258028268814087
Epoch  269  time  1.3025908470153809
Epoch  270  time  1.2533032894134521
Epoch  270  loss  0.825301132146157 correct 47
Epoch  271  time  1.252678632736206
Epoch  272  time  1.5587244033813477
Epoch  273  time  1.7726731300354004
Epoch  274  time  1.3115942478179932
Epoch  275  time  1.2549934387207031
Epoch  276  time  1.2657830715179443
Epoch  277  time  1.2558996677398682
Epoch  278  time  1.2741727828979492
Epoch  279  time  1.323850154876709
Epoch  280  time  1.2732925415039062
Epoch  280  loss  2.148447929616425 correct 46
Epoch  281  time  1.5332510471343994
Epoch  282  time  1.7902805805206299
Epoch  283  time  1.2510368824005127
Epoch  284  time  1.3090758323669434
Epoch  285  time  1.2686028480529785
Epoch  286  time  1.2494962215423584
Epoch  287  time  1.2623698711395264
Epoch  288  time  1.2635085582733154
Epoch  289  time  1.263138771057129
Epoch  290  time  1.5109853744506836
Epoch  290  loss  0.622147900562787 correct 48
Epoch  291  time  1.846221685409546
Epoch  292  time  1.2456941604614258
Epoch  293  time  1.2428514957427979
Epoch  294  time  1.2865185737609863
Epoch  295  time  1.3155097961425781
Epoch  296  time  1.2491707801818848
Epoch  297  time  1.2602760791778564
Epoch  298  time  1.2642364501953125
Epoch  299  time  1.3815836906433105
Epoch  300  time  1.9525928497314453
Epoch  300  loss  0.9529805360119229 correct 49
Epoch  301  time  1.2570407390594482
Epoch  302  time  1.2586004734039307
Epoch  303  time  1.2566962242126465
Epoch  304  time  1.266556978225708
Epoch  305  time  1.3048639297485352
Epoch  306  time  1.2498178482055664
Epoch  307  time  1.2743206024169922
Epoch  308  time  1.3795561790466309
Epoch  309  time  1.9450368881225586
Epoch  310  time  1.3077113628387451
Epoch  310  loss  1.0809809805221375 correct 49
Epoch  311  time  1.2613227367401123
Epoch  312  time  1.257812261581421
Epoch  313  time  1.2634716033935547
Epoch  314  time  1.3361022472381592
Epoch  315  time  1.2557368278503418
Epoch  316  time  1.2616662979125977
Epoch  317  time  1.364497423171997
Epoch  318  time  1.8714933395385742
Epoch  319  time  1.4358024597167969
Epoch  320  time  1.2432003021240234
Epoch  320  loss  0.8344846231915904 correct 49
Epoch  321  time  1.2439110279083252
Epoch  322  time  1.2658751010894775
Epoch  323  time  1.2723402976989746
Epoch  324  time  1.3296818733215332
Epoch  325  time  1.7241923809051514
Epoch  326  time  2.011401414871216
Epoch  327  time  1.7065520286560059
Epoch  328  time  1.254228115081787
Epoch  329  time  1.2675583362579346
Epoch  330  time  1.325622320175171
Epoch  330  loss  1.945323564000865 correct 50
Epoch  331  time  1.262134075164795
Epoch  332  time  1.246152400970459
Epoch  333  time  1.262801170349121
Epoch  334  time  1.2613742351531982
Epoch  335  time  1.6671233177185059
Epoch  336  time  1.7526278495788574
Epoch  337  time  1.2704224586486816
Epoch  338  time  1.2903499603271484
Epoch  339  time  1.2533185482025146
Epoch  340  time  1.3116071224212646
Epoch  340  loss  0.5197676283525035 correct 50
Epoch  341  time  1.2574355602264404
Epoch  342  time  1.2690258026123047
Epoch  343  time  1.247600793838501
Epoch  344  time  1.5172216892242432
Epoch  345  time  1.8503482341766357
Epoch  346  time  1.256535530090332
Epoch  347  time  1.2615411281585693
Epoch  348  time  1.2510697841644287
Epoch  349  time  1.248621940612793
Epoch  350  time  1.3502702713012695
Epoch  350  loss  0.028008420198538084 correct 49
Epoch  351  time  1.265045404434204
Epoch  352  time  1.2626469135284424
Epoch  353  time  1.506157398223877
Epoch  354  time  1.7964282035827637
Epoch  355  time  1.299774408340454
Epoch  356  time  1.2605361938476562
Epoch  357  time  1.2605903148651123
Epoch  358  time  1.25246000289917
Epoch  359  time  1.253427505493164
Epoch  360  time  1.3341288566589355
Epoch  360  loss  0.6640155866563755 correct 49
Epoch  361  time  1.2633991241455078
Epoch  362  time  1.483968734741211
Epoch  363  time  1.8232173919677734
Epoch  364  time  1.249779462814331
Epoch  365  time  1.3113577365875244
Epoch  366  time  1.2551698684692383
Epoch  367  time  1.2523977756500244
Epoch  368  time  1.252577543258667
Epoch  369  time  1.261885404586792
Epoch  370  time  1.310307264328003
Epoch  370  loss  0.5918029544970311 correct 49
Epoch  371  time  1.44309401512146
Epoch  372  time  1.8544127941131592
Epoch  373  time  1.2491941452026367
Epoch  374  time  1.2589237689971924
Epoch  375  time  1.3121631145477295
Epoch  376  time  1.2731423377990723
Epoch  377  time  1.2442805767059326
Epoch  378  time  1.2472641468048096
Epoch  379  time  1.265796422958374
Epoch  380  time  1.4608879089355469
Epoch  380  loss  1.001727144209916 correct 50
Epoch  381  time  1.8899929523468018
Epoch  382  time  1.2696375846862793
Epoch  383  time  1.2592017650604248
Epoch  384  time  1.2645542621612549
Epoch  385  time  1.3199260234832764
Epoch  386  time  1.2552196979522705
Epoch  387  time  1.2473478317260742
Epoch  388  time  1.2621190547943115
Epoch  389  time  1.3889074325561523
Epoch  390  time  1.9684569835662842
Epoch  390  loss  0.1747112976924862 correct 49
Epoch  391  time  1.3098242282867432
Epoch  392  time  1.2715511322021484
Epoch  393  time  1.2738676071166992
Epoch  394  time  1.3104586601257324
Epoch  395  time  1.3656587600708008
Epoch  396  time  1.2776496410369873
Epoch  397  time  1.2524147033691406
Epoch  398  time  1.427351474761963
Epoch  399  time  1.9342172145843506
Epoch  400  time  1.2817730903625488
Epoch  400  loss  0.4448953844364153 correct 49
Epoch  401  time  1.3141450881958008
Epoch  402  time  1.244640588760376
Epoch  403  time  1.2678091526031494
Epoch  404  time  1.2810797691345215
Epoch  405  time  1.2561862468719482
Epoch  406  time  1.320181131362915
Epoch  407  time  1.3893828392028809
Epoch  408  time  1.8803725242614746
Epoch  409  time  1.3076577186584473
Epoch  410  time  1.239243507385254
Epoch  410  loss  0.34938172839210035 correct 48
Epoch  411  time  1.3044493198394775
Epoch  412  time  1.2539989948272705
Epoch  413  time  1.2564454078674316
Epoch  414  time  1.2470042705535889
Epoch  415  time  1.2630834579467773
Epoch  416  time  1.3983080387115479
Epoch  417  time  1.8575661182403564
Epoch  418  time  1.382800579071045
Epoch  419  time  1.2551634311676025
Epoch  420  time  1.262446641921997
Epoch  420  loss  0.831832733810293 correct 49
Epoch  421  time  1.337906837463379
Epoch  422  time  1.250842571258545
Epoch  423  time  1.283182144165039
Epoch  424  time  1.2564318180084229
Epoch  425  time  1.2713594436645508
Epoch  426  time  1.926396369934082
Epoch  427  time  1.4383132457733154
Epoch  428  time  1.2599828243255615
Epoch  429  time  1.2528033256530762
Epoch  430  time  1.2511005401611328
Epoch  430  loss  1.2607670715527646 correct 49
Epoch  431  time  1.3142564296722412
Epoch  432  time  1.2645320892333984
Epoch  433  time  1.2564196586608887
Epoch  434  time  1.2533442974090576
Epoch  435  time  1.8111340999603271
Epoch  436  time  1.586338758468628
Epoch  437  time  1.2621874809265137
Epoch  438  time  1.2880654335021973
Epoch  439  time  1.2549424171447754
Epoch  440  time  1.255030870437622
Epoch  440  loss  0.57534888218695 correct 50
Epoch  441  time  1.318570613861084
Epoch  442  time  1.2511732578277588
Epoch  443  time  1.2501721382141113
Epoch  444  time  1.7492914199829102
Epoch  445  time  1.5356090068817139
Epoch  446  time  1.3071863651275635
Epoch  447  time  1.2461295127868652
Epoch  448  time  1.2738487720489502
Epoch  449  time  1.2504024505615234
Epoch  450  time  1.2700672149658203
Epoch  450  loss  0.5635230528560133 correct 48
Epoch  451  time  1.3262782096862793
Epoch  452  time  1.2568631172180176
Epoch  453  time  1.6918203830718994
Epoch  454  time  1.618879795074463
Epoch  455  time  1.2797911167144775
Epoch  456  time  1.3099539279937744
Epoch  457  time  1.2466232776641846
Epoch  458  time  1.2702367305755615
Epoch  459  time  1.268885612487793
Epoch  460  time  1.2621071338653564
Epoch  460  loss  0.07638390636934597 correct 49
Epoch  461  time  1.3029086589813232
Epoch  462  time  1.7003097534179688
Epoch  463  time  1.619978904724121
Epoch  464  time  1.2462716102600098
Epoch  465  time  1.262575387954712
Epoch  466  time  1.321366310119629
Epoch  467  time  1.2693054676055908
Epoch  468  time  1.2611238956451416
Epoch  469  time  1.2634212970733643
Epoch  470  time  1.2784931659698486
Epoch  470  loss  0.579196652021662 correct 50
Epoch  471  time  1.694612979888916
Epoch  472  time  1.655958652496338
Epoch  473  time  1.250185489654541
Epoch  474  time  1.2755458354949951
Epoch  475  time  1.2503776550292969
Epoch  476  time  1.3124134540557861
Epoch  477  time  1.2466285228729248
Epoch  478  time  1.2538776397705078
Epoch  479  time  1.2487266063690186
Epoch  480  time  1.529583215713501
Epoch  480  loss  0.40030843702080493 correct 49
Epoch  481  time  1.8063433170318604
Epoch  482  time  1.2649824619293213
Epoch  483  time  1.2527599334716797
Epoch  484  time  1.2534029483795166
Epoch  485  time  1.2450346946716309
Epoch  486  time  1.2560417652130127
Epoch  487  time  1.3027963638305664
Epoch  488  time  1.2603886127471924
Epoch  489  time  1.4735748767852783
Epoch  490  time  1.8215126991271973
Epoch  490  loss  1.0565769011519177 correct 49
Epoch  491  time  1.2356936931610107
Epoch  492  time  1.3454699516296387
Epoch  493  time  1.2541606426239014
Epoch  494  time  1.2481427192687988
Epoch  495  time  1.2435407638549805
Epoch  496  time  1.2672748565673828
Epoch  497  time  1.3175673484802246
Epoch  498  time  1.4195432662963867
Epoch  499  time  1.8735270500183105
```
</details>

python run_fast_tensor.py --BACKEND cpu --HIDDEN 100 --DATASET split --RATE 0.05

*Average time per epoch 0.17822088718414306 (for 500 epochs)*
<details>
<summary>View log</summary>
Epoch  0  time  5.812880277633667
Epoch  0  loss  6.299965442453414 correct 38
Epoch  1  time  0.16251111030578613
Epoch  2  time  0.1680619716644287
Epoch  3  time  0.1897268295288086
Epoch  4  time  0.15450191497802734
Epoch  5  time  0.16481900215148926
Epoch  6  time  0.15825915336608887
Epoch  7  time  0.19617915153503418
Epoch  8  time  0.1605088710784912
Epoch  9  time  0.1722400188446045
Epoch  10  time  0.16094589233398438
Epoch  10  loss  4.673394041927972 correct 43
Epoch  11  time  0.16105079650878906
Epoch  12  time  0.1621699333190918
Epoch  13  time  0.165510892868042
Epoch  14  time  0.16483569145202637
Epoch  15  time  0.17003321647644043
Epoch  16  time  0.16753196716308594
Epoch  17  time  0.18278193473815918
Epoch  18  time  0.16675400733947754
Epoch  19  time  0.16416192054748535
Epoch  20  time  0.1617891788482666
Epoch  20  loss  5.551104933050946 correct 42
Epoch  21  time  0.16971206665039062
Epoch  22  time  0.1632530689239502
Epoch  23  time  0.16401004791259766
Epoch  24  time  0.1634531021118164
Epoch  25  time  0.1614398956298828
Epoch  26  time  0.16529417037963867
Epoch  27  time  0.1649641990661621
Epoch  28  time  0.16439008712768555
Epoch  29  time  0.17525029182434082
Epoch  30  time  0.16326570510864258
Epoch  30  loss  4.578157306751618 correct 44
Epoch  31  time  0.16486811637878418
Epoch  32  time  0.1883080005645752
Epoch  33  time  0.1695718765258789
Epoch  34  time  0.16058897972106934
Epoch  35  time  0.16884803771972656
Epoch  36  time  0.1622328758239746
Epoch  37  time  0.16379594802856445
Epoch  38  time  0.16694188117980957
Epoch  39  time  0.16936874389648438
Epoch  40  time  0.16554880142211914
Epoch  40  loss  3.3049723306992274 correct 43
Epoch  41  time  0.1714649200439453
Epoch  42  time  0.16847991943359375
Epoch  43  time  0.16552400588989258
Epoch  44  time  0.16650700569152832
Epoch  45  time  0.17008686065673828
Epoch  46  time  0.17887330055236816
Epoch  47  time  0.16739988327026367
Epoch  48  time  0.16613411903381348
Epoch  49  time  0.16478800773620605
Epoch  50  time  0.16115498542785645
Epoch  50  loss  2.656477128712462 correct 47
Epoch  51  time  0.16986680030822754
Epoch  52  time  0.16801095008850098
Epoch  53  time  0.17088103294372559
Epoch  54  time  0.16724920272827148
Epoch  55  time  0.15935492515563965
Epoch  56  time  0.16775131225585938
Epoch  57  time  0.17178106307983398
Epoch  58  time  0.1811368465423584
Epoch  59  time  0.16319489479064941
Epoch  60  time  0.15975666046142578
Epoch  60  loss  2.309587407280773 correct 47
Epoch  61  time  0.16400384902954102
Epoch  62  time  0.25176000595092773
Epoch  63  time  0.17505788803100586
Epoch  64  time  0.16065192222595215
Epoch  65  time  0.16584515571594238
Epoch  66  time  0.17363405227661133
Epoch  67  time  0.16990113258361816
Epoch  68  time  0.1624600887298584
Epoch  69  time  0.16560792922973633
Epoch  70  time  0.16398882865905762
Epoch  70  loss  1.8048369723039732 correct 45
Epoch  71  time  0.16438508033752441
Epoch  72  time  0.1669309139251709
Epoch  73  time  0.1584639549255371
Epoch  74  time  0.16217589378356934
Epoch  75  time  0.1799759864807129
Epoch  76  time  0.16730093955993652
Epoch  77  time  0.168076753616333
Epoch  78  time  0.16747713088989258
Epoch  79  time  0.16014504432678223
Epoch  80  time  0.16531682014465332
Epoch  80  loss  1.4781938393520389 correct 48
Epoch  81  time  0.16383719444274902
Epoch  82  time  0.1658649444580078
Epoch  83  time  0.16495084762573242
Epoch  84  time  0.16400408744812012
Epoch  85  time  0.16027498245239258
Epoch  86  time  0.15755486488342285
Epoch  87  time  0.17409729957580566
Epoch  88  time  0.17308998107910156
Epoch  89  time  0.16569805145263672
Epoch  90  time  0.17068886756896973
Epoch  90  loss  1.1568680807250202 correct 50
Epoch  91  time  0.16659116744995117
Epoch  92  time  0.16870403289794922
Epoch  93  time  0.16960787773132324
Epoch  94  time  0.17169189453125
Epoch  95  time  0.15613007545471191
Epoch  96  time  0.17171573638916016
Epoch  97  time  0.16569185256958008
Epoch  98  time  0.1645500659942627
Epoch  99  time  0.1601099967956543
Epoch  100  time  0.16482996940612793
Epoch  100  loss  2.361816133196858 correct 49
Epoch  101  time  0.16564297676086426
Epoch  102  time  0.1613161563873291
Epoch  103  time  0.1705951690673828
Epoch  104  time  0.16111993789672852
Epoch  105  time  0.1713709831237793
Epoch  106  time  0.16398119926452637
Epoch  107  time  0.16479897499084473
Epoch  108  time  0.16149210929870605
Epoch  109  time  0.17169404029846191
Epoch  110  time  0.16050004959106445
Epoch  110  loss  2.24042670157423 correct 47
Epoch  111  time  0.16513586044311523
Epoch  112  time  0.16773366928100586
Epoch  113  time  0.16391897201538086
Epoch  114  time  0.1699509620666504
Epoch  115  time  0.15961694717407227
Epoch  116  time  0.1657111644744873
Epoch  117  time  0.17626023292541504
Epoch  118  time  0.16486907005310059
Epoch  119  time  0.16315507888793945
Epoch  120  time  0.16549110412597656
Epoch  120  loss  2.2216795781326923 correct 50
Epoch  121  time  0.16085195541381836
Epoch  122  time  0.16089820861816406
Epoch  123  time  0.1684730052947998
Epoch  124  time  0.17455792427062988
Epoch  125  time  0.1739950180053711
Epoch  126  time  0.16737890243530273
Epoch  127  time  0.17258000373840332
Epoch  128  time  0.1606290340423584
Epoch  129  time  0.16995501518249512
Epoch  130  time  0.16884303092956543
Epoch  130  loss  1.1379019850664214 correct 47
Epoch  131  time  0.16127300262451172
Epoch  132  time  0.18788981437683105
Epoch  133  time  0.1738591194152832
Epoch  134  time  0.1670997142791748
Epoch  135  time  0.16625571250915527
Epoch  136  time  0.16292405128479004
Epoch  137  time  0.16231894493103027
Epoch  138  time  0.16090083122253418
Epoch  139  time  0.16729235649108887
Epoch  140  time  0.15700197219848633
Epoch  140  loss  1.5521199200458737 correct 49
Epoch  141  time  0.16013097763061523
Epoch  142  time  0.16614413261413574
Epoch  143  time  0.16914987564086914
Epoch  144  time  0.16234922409057617
Epoch  145  time  0.16066217422485352
Epoch  146  time  0.16094326972961426
Epoch  147  time  0.17560386657714844
Epoch  148  time  0.16215991973876953
Epoch  149  time  0.16127920150756836
Epoch  150  time  0.16065669059753418
Epoch  150  loss  0.0747105758651076 correct 49
Epoch  151  time  0.16588091850280762
Epoch  152  time  0.1657569408416748
Epoch  153  time  0.16190695762634277
Epoch  154  time  0.15941691398620605
Epoch  155  time  0.1660170555114746
Epoch  156  time  0.16386723518371582
Epoch  157  time  0.16389727592468262
Epoch  158  time  0.16022086143493652
Epoch  159  time  0.16182994842529297
Epoch  160  time  0.16765809059143066
Epoch  160  loss  0.42305380514302926 correct 49
Epoch  161  time  0.16209793090820312
Epoch  162  time  0.16727709770202637
Epoch  163  time  0.1655750274658203
Epoch  164  time  0.16562795639038086
Epoch  165  time  0.16141080856323242
Epoch  166  time  0.16812491416931152
Epoch  167  time  0.16246604919433594
Epoch  168  time  0.16309714317321777
Epoch  169  time  0.16057801246643066
Epoch  170  time  0.1633920669555664
Epoch  170  loss  0.8828645842049858 correct 49
Epoch  171  time  0.16428184509277344
Epoch  172  time  0.16390419006347656
Epoch  173  time  0.1613328456878662
Epoch  174  time  0.1708071231842041
Epoch  175  time  0.1599879264831543
Epoch  176  time  0.16424083709716797
Epoch  177  time  0.17495131492614746
Epoch  178  time  0.16982793807983398
Epoch  179  time  0.16135597229003906
Epoch  180  time  0.16578102111816406
Epoch  180  loss  0.753774778350546 correct 47
Epoch  181  time  0.15842199325561523
Epoch  182  time  0.1719522476196289
Epoch  183  time  0.16481709480285645
Epoch  184  time  0.16399788856506348
Epoch  185  time  0.16988587379455566
Epoch  186  time  0.16211199760437012
Epoch  187  time  0.15711593627929688
Epoch  188  time  0.17282509803771973
Epoch  189  time  0.16498780250549316
Epoch  190  time  0.16140389442443848
Epoch  190  loss  1.8289457900056898 correct 49
Epoch  191  time  0.17498517036437988
Epoch  192  time  0.15912675857543945
Epoch  193  time  0.16917920112609863
Epoch  194  time  0.168503999710083
Epoch  195  time  0.18264293670654297
Epoch  196  time  0.16407990455627441
Epoch  197  time  0.165802001953125
Epoch  198  time  0.16225099563598633
Epoch  199  time  0.16587018966674805
Epoch  200  time  0.16391611099243164
Epoch  200  loss  1.0668349009894857 correct 50
Epoch  201  time  0.17189502716064453
Epoch  202  time  0.1656639575958252
Epoch  203  time  0.16740131378173828
Epoch  204  time  0.16240882873535156
Epoch  205  time  0.16498398780822754
Epoch  206  time  0.17597413063049316
Epoch  207  time  0.16465401649475098
Epoch  208  time  0.16617298126220703
Epoch  209  time  0.19770598411560059
Epoch  210  time  0.1628742218017578
Epoch  210  loss  1.228131263816941 correct 50
Epoch  211  time  0.1640608310699463
Epoch  212  time  0.17288804054260254
Epoch  213  time  0.17136502265930176
Epoch  214  time  0.16716599464416504
Epoch  215  time  0.162261962890625
Epoch  216  time  0.1660010814666748
Epoch  217  time  0.16603493690490723
Epoch  218  time  0.16542911529541016
Epoch  219  time  0.19281721115112305
Epoch  220  time  0.1625990867614746
Epoch  220  loss  2.137146314446272 correct 48
Epoch  221  time  0.16604399681091309
Epoch  222  time  0.16739106178283691
Epoch  223  time  0.16587209701538086
Epoch  224  time  0.16601300239562988
Epoch  225  time  0.16282391548156738
Epoch  226  time  0.1649000644683838
Epoch  227  time  0.17114496231079102
Epoch  228  time  0.16376614570617676
Epoch  229  time  0.16905689239501953
Epoch  230  time  0.164963960647583
Epoch  230  loss  2.1656003155620995 correct 49
Epoch  231  time  0.16399478912353516
Epoch  232  time  0.16591119766235352
Epoch  233  time  0.1640031337738037
Epoch  234  time  0.16529393196105957
Epoch  235  time  0.16241192817687988
Epoch  236  time  0.16888189315795898
Epoch  237  time  0.17117810249328613
Epoch  238  time  0.169327974319458
Epoch  239  time  0.16468501091003418
Epoch  240  time  0.16609525680541992
Epoch  240  loss  0.028465333928636264 correct 49
Epoch  241  time  0.1643049716949463
Epoch  242  time  0.17069005966186523
Epoch  243  time  0.16249489784240723
Epoch  244  time  0.1708371639251709
Epoch  245  time  0.17694616317749023
Epoch  246  time  0.17272615432739258
Epoch  247  time  0.16568708419799805
Epoch  248  time  0.1636826992034912
Epoch  249  time  0.16533994674682617
Epoch  250  time  0.1685478687286377
Epoch  250  loss  0.18980090350227385 correct 50
Epoch  251  time  0.16513299942016602
Epoch  252  time  0.174940824508667
Epoch  253  time  0.16910791397094727
Epoch  254  time  0.1634361743927002
Epoch  255  time  0.1647930145263672
Epoch  256  time  0.15709900856018066
Epoch  257  time  0.1612250804901123
Epoch  258  time  0.16815900802612305
Epoch  259  time  0.16346001625061035
Epoch  260  time  0.17081904411315918
Epoch  260  loss  0.3110870748604478 correct 50
Epoch  261  time  0.16373085975646973
Epoch  262  time  0.16162610054016113
Epoch  263  time  0.16382694244384766
Epoch  264  time  0.16470003128051758
Epoch  265  time  0.17461299896240234
Epoch  266  time  0.18626189231872559
Epoch  267  time  0.1621718406677246
Epoch  268  time  0.17345380783081055
Epoch  269  time  0.16823101043701172
Epoch  270  time  0.16838598251342773
Epoch  270  loss  0.5021419075395801 correct 50
Epoch  271  time  0.18379998207092285
Epoch  272  time  0.16713595390319824
Epoch  273  time  0.16523075103759766
Epoch  274  time  0.17580723762512207
Epoch  275  time  0.179764986038208
Epoch  276  time  0.16394400596618652
Epoch  277  time  0.17062711715698242
Epoch  278  time  0.16796588897705078
Epoch  279  time  0.15674901008605957
Epoch  280  time  0.16502761840820312
Epoch  280  loss  0.638999496853944 correct 50
Epoch  281  time  0.17049384117126465
Epoch  282  time  0.16442418098449707
Epoch  283  time  0.1715257167816162
Epoch  284  time  0.16379690170288086
Epoch  285  time  0.16781210899353027
Epoch  286  time  0.16264772415161133
Epoch  287  time  0.1632399559020996
Epoch  288  time  0.16524696350097656
Epoch  289  time  0.16897010803222656
Epoch  290  time  0.17293167114257812
Epoch  290  loss  0.4018998247987849 correct 50
Epoch  291  time  0.16000008583068848
Epoch  292  time  0.16869211196899414
Epoch  293  time  0.18726825714111328
Epoch  294  time  0.18021321296691895
Epoch  295  time  0.16704702377319336
Epoch  296  time  0.1621391773223877
Epoch  297  time  0.16463685035705566
Epoch  298  time  0.16535711288452148
Epoch  299  time  0.1636488437652588
Epoch  300  time  0.1708049774169922
Epoch  300  loss  0.432342184672106 correct 49
Epoch  301  time  0.16086196899414062
Epoch  302  time  0.1650862693786621
Epoch  303  time  0.1610250473022461
Epoch  304  time  0.16022920608520508
Epoch  305  time  0.16840481758117676
Epoch  306  time  0.16434001922607422
Epoch  307  time  0.16614818572998047
Epoch  308  time  0.1672229766845703
Epoch  309  time  0.16260886192321777
Epoch  310  time  0.1583878993988037
Epoch  310  loss  0.7951816329492638 correct 50
Epoch  311  time  0.16870403289794922
Epoch  312  time  0.16640877723693848
Epoch  313  time  0.16821694374084473
Epoch  314  time  0.16650891304016113
Epoch  315  time  0.16810011863708496
Epoch  316  time  0.15980815887451172
Epoch  317  time  0.16176819801330566
Epoch  318  time  0.16193509101867676
Epoch  319  time  0.16740989685058594
Epoch  320  time  0.1691751480102539
Epoch  320  loss  0.14649540505643657 correct 49
Epoch  321  time  0.16274690628051758
Epoch  322  time  0.16390085220336914
Epoch  323  time  0.1674351692199707
Epoch  324  time  0.17423486709594727
Epoch  325  time  0.1711890697479248
Epoch  326  time  0.16433215141296387
Epoch  327  time  0.16266322135925293
Epoch  328  time  0.15913009643554688
Epoch  329  time  0.16708683967590332
Epoch  330  time  0.1682732105255127
Epoch  330  loss  0.7053255771137479 correct 50
Epoch  331  time  0.16910099983215332
Epoch  332  time  0.16431307792663574
Epoch  333  time  0.1649460792541504
Epoch  334  time  0.1663801670074463
Epoch  335  time  0.16372203826904297
Epoch  336  time  0.16646194458007812
Epoch  337  time  0.1668398380279541
Epoch  338  time  0.16350889205932617
Epoch  339  time  0.16452670097351074
Epoch  340  time  0.1619279384613037
Epoch  340  loss  0.06949610802782903 correct 49
Epoch  341  time  0.1653590202331543
Epoch  342  time  0.16254305839538574
Epoch  343  time  0.16850614547729492
Epoch  344  time  0.16942906379699707
Epoch  345  time  0.1625969409942627
Epoch  346  time  0.1746659278869629
Epoch  347  time  0.1647341251373291
Epoch  348  time  0.1638178825378418
Epoch  349  time  0.16263508796691895
Epoch  350  time  0.15831804275512695
Epoch  350  loss  0.17017203053158733 correct 50
Epoch  351  time  0.16475486755371094
Epoch  352  time  0.16182327270507812
Epoch  353  time  0.17169690132141113
Epoch  354  time  0.17544102668762207
Epoch  355  time  0.1649918556213379
Epoch  356  time  0.17066383361816406
Epoch  357  time  0.1688370704650879
Epoch  358  time  0.16911005973815918
Epoch  359  time  0.16337108612060547
Epoch  360  time  0.17214417457580566
Epoch  360  loss  0.09595955918254187 correct 49
Epoch  361  time  0.16233301162719727
Epoch  362  time  0.1695549488067627
Epoch  363  time  0.16603612899780273
Epoch  364  time  0.16926002502441406
Epoch  365  time  0.16003990173339844
Epoch  366  time  0.15941190719604492
Epoch  367  time  0.1609337329864502
Epoch  368  time  0.16385483741760254
Epoch  369  time  0.16630125045776367
Epoch  370  time  0.17153692245483398
Epoch  370  loss  0.458780272016884 correct 50
Epoch  371  time  0.16430997848510742
Epoch  372  time  0.18633008003234863
Epoch  373  time  0.16205692291259766
Epoch  374  time  0.1690049171447754
Epoch  375  time  0.1664748191833496
Epoch  376  time  0.17229199409484863
Epoch  377  time  0.16955995559692383
Epoch  378  time  0.15726613998413086
Epoch  379  time  0.15936732292175293
Epoch  380  time  0.16168808937072754
Epoch  380  loss  0.42172221884965605 correct 50
Epoch  381  time  0.16480612754821777
Epoch  382  time  0.16620206832885742
Epoch  383  time  0.17281293869018555
Epoch  384  time  0.1614820957183838
Epoch  385  time  0.16449213027954102
Epoch  386  time  0.18870806694030762
Epoch  387  time  0.16651487350463867
Epoch  388  time  0.16819500923156738
Epoch  389  time  0.17091679573059082
Epoch  390  time  0.1616220474243164
Epoch  390  loss  0.5009119948151384 correct 49
Epoch  391  time  0.1632399559020996
Epoch  392  time  0.16445493698120117
Epoch  393  time  0.16516709327697754
Epoch  394  time  0.16602802276611328
Epoch  395  time  0.1670970916748047
Epoch  396  time  0.16903901100158691
Epoch  397  time  0.20412588119506836
Epoch  398  time  0.16910910606384277
Epoch  399  time  0.17916274070739746
Epoch  400  time  0.16525602340698242
Epoch  400  loss  0.34710264295458876 correct 49
Epoch  401  time  0.16665863990783691
Epoch  402  time  0.16095781326293945
Epoch  403  time  0.1616971492767334
Epoch  404  time  0.17180585861206055
Epoch  405  time  0.1698131561279297
Epoch  406  time  0.17052221298217773
Epoch  407  time  0.15920495986938477
Epoch  408  time  0.16699481010437012
Epoch  409  time  0.16488409042358398
Epoch  410  time  0.17419886589050293
Epoch  410  loss  1.083017806069822 correct 50
Epoch  411  time  0.16489076614379883
Epoch  412  time  0.178879976272583
Epoch  413  time  0.16595792770385742
Epoch  414  time  0.16579389572143555
Epoch  415  time  0.1647946834564209
Epoch  416  time  0.1803150177001953
Epoch  417  time  0.16048598289489746
Epoch  418  time  0.1608738899230957
Epoch  419  time  0.17108488082885742
Epoch  420  time  0.17149972915649414
Epoch  420  loss  0.7837735209384745 correct 49
Epoch  421  time  0.16498923301696777
Epoch  422  time  0.1676499843597412
Epoch  423  time  0.16995596885681152
Epoch  424  time  0.16092705726623535
Epoch  425  time  0.16416573524475098
Epoch  426  time  0.16509485244750977
Epoch  427  time  0.15676522254943848
Epoch  428  time  0.16811490058898926
Epoch  429  time  0.16418099403381348
Epoch  430  time  0.16176199913024902
Epoch  430  loss  0.3433921184790791 correct 50
Epoch  431  time  0.1628100872039795
Epoch  432  time  0.16505002975463867
Epoch  433  time  0.16216588020324707
Epoch  434  time  0.1593480110168457
Epoch  435  time  0.1692519187927246
Epoch  436  time  0.1646709442138672
Epoch  437  time  0.16933512687683105
Epoch  438  time  0.1672060489654541
Epoch  439  time  0.16522884368896484
Epoch  440  time  0.1754469871520996
Epoch  440  loss  0.9641081439314343 correct 49
Epoch  441  time  0.16523289680480957
Epoch  442  time  0.1741199493408203
Epoch  443  time  0.16635704040527344
Epoch  444  time  0.18615293502807617
Epoch  445  time  0.16755986213684082
Epoch  446  time  0.1676950454711914
Epoch  447  time  0.16113901138305664
Epoch  448  time  0.16155314445495605
Epoch  449  time  0.16271615028381348
Epoch  450  time  0.16478466987609863
Epoch  450  loss  0.28996715561643377 correct 50
Epoch  451  time  0.16081738471984863
Epoch  452  time  0.16526103019714355
Epoch  453  time  0.1660470962524414
Epoch  454  time  0.18726110458374023
Epoch  455  time  0.17027902603149414
Epoch  456  time  0.1661698818206787
Epoch  457  time  0.17485594749450684
Epoch  458  time  0.16444706916809082
Epoch  459  time  0.1707298755645752
Epoch  460  time  0.16479897499084473
Epoch  460  loss  0.2109341256935902 correct 49
Epoch  461  time  0.16462230682373047
Epoch  462  time  0.17879295349121094
Epoch  463  time  0.16889405250549316
Epoch  464  time  0.17074012756347656
Epoch  465  time  0.16207313537597656
Epoch  466  time  0.1690502166748047
Epoch  467  time  0.16461515426635742
Epoch  468  time  0.17069482803344727
Epoch  469  time  0.1631157398223877
Epoch  470  time  0.16763687133789062
Epoch  470  loss  0.2199547126823806 correct 50
Epoch  471  time  0.17697811126708984
Epoch  472  time  0.16497588157653809
Epoch  473  time  0.15984892845153809
Epoch  474  time  0.16507887840270996
Epoch  475  time  0.1642298698425293
Epoch  476  time  0.16721010208129883
Epoch  477  time  0.17312121391296387
Epoch  478  time  0.16267704963684082
Epoch  479  time  0.16689586639404297
Epoch  480  time  0.16540002822875977
Epoch  480  loss  0.8232499029831767 correct 49
Epoch  481  time  0.16593503952026367
Epoch  482  time  0.16967511177062988
Epoch  483  time  0.17233490943908691
Epoch  484  time  0.16360926628112793
Epoch  485  time  0.16111207008361816
Epoch  486  time  0.16167402267456055
Epoch  487  time  0.16358709335327148
Epoch  488  time  0.1584010124206543
Epoch  489  time  0.16306209564208984
Epoch  490  time  0.1683502197265625
Epoch  490  loss  0.5775752494133057 correct 50
Epoch  491  time  0.17284297943115234
Epoch  492  time  0.15873289108276367
Epoch  493  time  0.16085505485534668
Epoch  494  time  0.16878271102905273
Epoch  495  time  0.16488027572631836
Epoch  496  time  0.15981602668762207
Epoch  497  time  0.16437697410583496
Epoch  498  time  0.16581392288208008
Epoch  499  time  0.17302703857421875
</details>

## Simple Dataset

python3.12 project/run_fast_tensor.py --BACKEND gpu --HIDDEN 100 --DATASET simple --RATE 0.05

*Average time per epoch 1.3617659120559693 (for 500 epochs)*

<details>
  <summary>View log</summary>
```
Epoch  0  loss  8.517706358137016 correct 38
Epoch  1  time  1.2621335983276367
Epoch  2  time  1.3114116191864014
Epoch  3  time  1.8280744552612305
Epoch  4  time  1.4922733306884766
Epoch  5  time  1.2727456092834473
Epoch  6  time  1.3141729831695557
Epoch  7  time  1.3220298290252686
Epoch  8  time  1.2452688217163086
Epoch  9  time  1.2576797008514404
Epoch  10  time  1.2530860900878906
Epoch  10  loss  1.7427835114322234 correct 50
Epoch  11  time  1.2688202857971191
Epoch  12  time  1.844618558883667
Epoch  13  time  1.4776864051818848
Epoch  14  time  1.251392126083374
Epoch  15  time  1.270171880722046
Epoch  16  time  1.2559845447540283
Epoch  17  time  1.329521894454956
Epoch  18  time  1.267516851425171
Epoch  19  time  1.278723955154419
Epoch  20  time  1.2559785842895508
Epoch  20  loss  1.9245063403993663 correct 49
Epoch  21  time  1.7636797428131104
Epoch  22  time  1.588059425354004
Epoch  23  time  1.2660770416259766
Epoch  24  time  1.284989833831787
Epoch  25  time  1.2783291339874268
Epoch  26  time  1.3122639656066895
Epoch  27  time  1.3244917392730713
Epoch  28  time  1.299133539199829
Epoch  29  time  1.266143560409546
Epoch  30  time  1.8021447658538818
Epoch  30  loss  1.4329459510388098 correct 50
Epoch  31  time  1.496931791305542
Epoch  32  time  1.3193867206573486
Epoch  33  time  1.2613325119018555
Epoch  34  time  1.257277488708496
Epoch  35  time  1.2651009559631348
Epoch  36  time  1.2622911930084229
Epoch  37  time  1.3390579223632812
Epoch  38  time  1.2563591003417969
Epoch  39  time  1.8240759372711182
Epoch  40  time  1.5352458953857422
Epoch  40  loss  0.9092975707689611 correct 50
Epoch  41  time  1.2504675388336182
Epoch  42  time  1.3218412399291992
Epoch  43  time  1.253749132156372
Epoch  44  time  1.2559635639190674
Epoch  45  time  1.2571797370910645
Epoch  46  time  1.2697761058807373
Epoch  47  time  1.3480861186981201
Epoch  48  time  1.7651622295379639
Epoch  49  time  1.5465188026428223
Epoch  50  time  1.288114070892334
Epoch  50  loss  0.562372789990172 correct 50
Epoch  51  time  1.2489097118377686
Epoch  52  time  1.3019344806671143
Epoch  53  time  1.2717516422271729
Epoch  54  time  1.2936477661132812
Epoch  55  time  1.2654409408569336
Epoch  56  time  1.2472670078277588
Epoch  57  time  1.765758752822876
Epoch  58  time  1.5913591384887695
Epoch  59  time  1.259366750717163
Epoch  60  time  1.267850399017334
Epoch  60  loss  0.9640740371685431 correct 50
Epoch  61  time  1.297175407409668
Epoch  62  time  1.3283653259277344
Epoch  63  time  1.2811620235443115
Epoch  64  time  1.2710132598876953
Epoch  65  time  1.2617406845092773
Epoch  66  time  1.7113218307495117
Epoch  67  time  1.6179604530334473
Epoch  68  time  1.3519916534423828
Epoch  69  time  1.2606050968170166
Epoch  70  time  1.2463021278381348
Epoch  70  loss  0.4290298606585699 correct 50
Epoch  71  time  1.2553248405456543
Epoch  72  time  1.2994239330291748
Epoch  73  time  1.2975788116455078
Epoch  74  time  1.2598114013671875
Epoch  75  time  1.6952474117279053
Epoch  76  time  1.6558542251586914
Epoch  77  time  1.2597968578338623
Epoch  78  time  1.31815505027771
Epoch  79  time  1.256415605545044
Epoch  80  time  1.2538397312164307
Epoch  80  loss  0.09251756266279135 correct 50
Epoch  81  time  1.24580979347229
Epoch  82  time  1.265918493270874
Epoch  83  time  1.3255808353424072
Epoch  84  time  1.6103246212005615
Epoch  85  time  1.6671741008758545
Epoch  86  time  1.252532958984375
Epoch  87  time  1.261810541152954
Epoch  88  time  1.298978328704834
Epoch  89  time  1.2458512783050537
Epoch  90  time  1.2704951763153076
Epoch  90  loss  0.10793910416629839 correct 50
Epoch  91  time  1.277205228805542
Epoch  92  time  1.2508997917175293
Epoch  93  time  1.6250009536743164
Epoch  94  time  1.756829023361206
Epoch  95  time  1.266977071762085
Epoch  96  time  1.2560245990753174
Epoch  97  time  1.258866310119629
Epoch  98  time  1.3181371688842773
Epoch  99  time  1.2524549961090088
Epoch  100  time  1.246438980102539
Epoch  100  loss  0.1746224534892563 correct 50
Epoch  101  time  1.2556047439575195
Epoch  102  time  1.5102510452270508
Epoch  103  time  1.832627773284912
Epoch  104  time  1.241358757019043
Epoch  105  time  1.2640454769134521
Epoch  106  time  1.2582306861877441
Epoch  107  time  1.2558410167694092
Epoch  108  time  1.308762550354004
Epoch  109  time  1.2510337829589844
Epoch  110  time  1.256777048110962
Epoch  110  loss  0.1812401240529424 correct 50
Epoch  111  time  1.4431865215301514
Epoch  112  time  1.8335106372833252
Epoch  113  time  1.3159902095794678
Epoch  114  time  1.243912935256958
Epoch  115  time  1.2405047416687012
Epoch  116  time  1.2732086181640625
Epoch  117  time  1.2461497783660889
Epoch  118  time  1.2913267612457275
Epoch  119  time  1.2531297206878662
Epoch  120  time  1.3531222343444824
Epoch  120  loss  0.21806659302094045 correct 50
Epoch  121  time  1.8691630363464355
Epoch  122  time  1.3088579177856445
Epoch  123  time  1.2352485656738281
Epoch  124  time  1.314241886138916
Epoch  125  time  1.2718942165374756
Epoch  126  time  1.248436450958252
Epoch  127  time  1.2530550956726074
Epoch  128  time  1.2681705951690674
Epoch  129  time  1.362729787826538
Epoch  130  time  1.8191795349121094
Epoch  130  loss  0.2391477764899231 correct 50
Epoch  131  time  1.3808414936065674
Epoch  132  time  1.251378059387207
Epoch  133  time  1.2457575798034668
Epoch  134  time  1.3011314868927002
Epoch  135  time  1.3068854808807373
Epoch  136  time  1.2559900283813477
Epoch  137  time  1.25205397605896
Epoch  138  time  1.251209020614624
Epoch  139  time  1.8561830520629883
Epoch  140  time  1.5634765625
Epoch  140  loss  0.05045370410289718 correct 50
Epoch  141  time  1.2588272094726562
Epoch  142  time  1.5624134540557861
Epoch  143  time  1.771824598312378
Epoch  144  time  1.2956902980804443
Epoch  145  time  1.264214038848877
Epoch  146  time  1.2536380290985107
Epoch  147  time  1.518477439880371
Epoch  148  time  1.790679693222046
Epoch  149  time  1.3400015830993652
Epoch  150  time  1.2540497779846191
Epoch  150  loss  0.41706523148305646 correct 50
Epoch  151  time  1.2585272789001465
Epoch  152  time  1.2500596046447754
Epoch  153  time  1.2451987266540527
Epoch  154  time  1.3180079460144043
Epoch  155  time  1.2804827690124512
Epoch  156  time  1.4876863956451416
Epoch  157  time  1.818713903427124
Epoch  158  time  1.2511277198791504
Epoch  159  time  1.3155889511108398
Epoch  160  time  1.2923967838287354
Epoch  160  loss  0.5426695877528535 correct 50
Epoch  161  time  1.2495224475860596
Epoch  162  time  1.2452757358551025
Epoch  163  time  1.2556793689727783
Epoch  164  time  1.3121700286865234
Epoch  165  time  1.4662694931030273
Epoch  166  time  1.8290140628814697
Epoch  167  time  1.2516393661499023
Epoch  168  time  1.246872901916504
Epoch  169  time  1.2935895919799805
Epoch  170  time  1.2601375579833984
Epoch  170  loss  0.02312524816504076 correct 50
Epoch  171  time  1.2515087127685547
Epoch  172  time  1.2587220668792725
Epoch  173  time  1.258636713027954
Epoch  174  time  1.4956867694854736
Epoch  175  time  1.8727428913116455
Epoch  176  time  1.247922658920288
Epoch  177  time  1.2432310581207275
Epoch  178  time  1.2632660865783691
Epoch  179  time  1.3165135383605957
Epoch  180  time  1.2471179962158203
Epoch  180  loss  0.1559836459758614 correct 50
Epoch  181  time  1.2523434162139893
Epoch  182  time  1.2774667739868164
Epoch  183  time  1.3383054733276367
Epoch  184  time  1.9850733280181885
Epoch  185  time  1.2928450107574463
Epoch  186  time  1.2535011768341064
Epoch  187  time  1.2631986141204834
Epoch  188  time  1.2473342418670654
Epoch  189  time  1.3079447746276855
Epoch  190  time  1.257845163345337
Epoch  190  loss  0.15081393382533745 correct 50
Epoch  191  time  1.2464418411254883
Epoch  192  time  1.3029398918151855
Epoch  193  time  1.8613338470458984
Epoch  194  time  1.4808330535888672
Epoch  195  time  1.259587049484253
Epoch  196  time  1.2684285640716553
Epoch  197  time  1.253765344619751
Epoch  198  time  1.2694127559661865
Epoch  199  time  1.2965662479400635
Epoch  200  time  1.2443373203277588
Epoch  200  loss  0.19907916246596263 correct 50
Epoch  201  time  1.2622215747833252
Epoch  202  time  1.84071683883667
Epoch  203  time  1.4537749290466309
Epoch  204  time  1.349245548248291
Epoch  205  time  1.259962797164917
Epoch  206  time  1.2582640647888184
Epoch  207  time  1.2520217895507812
Epoch  208  time  1.2601289749145508
Epoch  209  time  1.327502727508545
Epoch  210  time  1.2564725875854492
Epoch  210  loss  0.24104216799028888 correct 50
Epoch  211  time  1.8038206100463867
Epoch  212  time  1.4837291240692139
Epoch  213  time  1.2614176273345947
Epoch  214  time  1.3224704265594482
Epoch  215  time  1.2536396980285645
Epoch  216  time  1.250572919845581
Epoch  217  time  1.2816319465637207
Epoch  218  time  1.2574195861816406
Epoch  219  time  1.3161499500274658
Epoch  220  time  1.750532627105713
Epoch  220  loss  0.16931315642988107 correct 50
Epoch  221  time  1.5274393558502197
Epoch  222  time  1.283172369003296
Epoch  223  time  1.2452936172485352
Epoch  224  time  1.3104000091552734
Epoch  225  time  1.2497367858886719
Epoch  226  time  1.2940306663513184
Epoch  227  time  1.2434966564178467
Epoch  228  time  1.2558784484863281
Epoch  229  time  1.7466096878051758
Epoch  230  time  1.6105003356933594
Epoch  230  loss  0.07876092341195576 correct 50
Epoch  231  time  1.2750873565673828
Epoch  232  time  1.2526600360870361
Epoch  233  time  1.250244379043579
Epoch  234  time  1.3041749000549316
Epoch  235  time  1.2436718940734863
Epoch  236  time  1.2574808597564697
Epoch  237  time  1.2543954849243164
Epoch  238  time  1.5835614204406738
Epoch  239  time  1.777756929397583
Epoch  240  time  1.2381491661071777
Epoch  240  loss  0.022595182511906755 correct 50
Epoch  241  time  1.2608745098114014
Epoch  242  time  1.249096393585205
Epoch  243  time  1.2445731163024902
Epoch  244  time  1.3066308498382568
Epoch  245  time  1.2620558738708496
Epoch  246  time  1.2687957286834717
Epoch  247  time  1.5477828979492188
Epoch  248  time  1.7991225719451904
Epoch  249  time  1.2975342273712158
Epoch  250  time  1.25496506690979
Epoch  250  loss  0.10426080655641823 correct 50
Epoch  251  time  1.2439937591552734
Epoch  252  time  1.2416272163391113
Epoch  253  time  1.2927100658416748
Epoch  254  time  1.3297438621520996
Epoch  255  time  1.243105173110962
Epoch  256  time  1.484811544418335
Epoch  257  time  1.80507493019104
Epoch  258  time  1.2653679847717285
Epoch  259  time  1.3006305694580078
Epoch  260  time  1.2555015087127686
Epoch  260  loss  0.1201441194340816 correct 50
Epoch  261  time  1.2577548027038574
Epoch  262  time  1.265655517578125
Epoch  263  time  1.2497105598449707
Epoch  264  time  1.307967185974121
Epoch  265  time  1.4307727813720703
Epoch  266  time  1.862553358078003
Epoch  267  time  1.2463469505310059
Epoch  268  time  1.2432348728179932
Epoch  269  time  1.3184239864349365
Epoch  270  time  1.2730579376220703
Epoch  270  loss  0.18517709786105419 correct 50
Epoch  271  time  1.2498412132263184
Epoch  272  time  1.2472152709960938
Epoch  273  time  1.2689967155456543
Epoch  274  time  1.466672658920288
Epoch  275  time  1.888744592666626
Epoch  276  time  1.2648169994354248
Epoch  277  time  1.2556588649749756
Epoch  278  time  1.2536373138427734
Epoch  279  time  1.309136152267456
Epoch  280  time  1.2558872699737549
Epoch  280  loss  0.0928961983208578 correct 50
Epoch  281  time  1.2668092250823975
Epoch  282  time  1.2592415809631348
Epoch  283  time  1.3770110607147217
Epoch  284  time  1.9559576511383057
Epoch  285  time  1.2863216400146484
Epoch  286  time  1.2532520294189453
Epoch  287  time  1.2594351768493652
Epoch  288  time  1.255664348602295
Epoch  289  time  1.2470324039459229
Epoch  290  time  1.3392143249511719
Epoch  290  loss  0.20284714557512315 correct 50
Epoch  291  time  1.2477622032165527
Epoch  292  time  1.3438608646392822
Epoch  293  time  1.8979272842407227
Epoch  294  time  1.3556139469146729
Epoch  295  time  1.3035292625427246
Epoch  296  time  1.247471809387207
Epoch  297  time  1.249516487121582
Epoch  298  time  1.2481465339660645
Epoch  299  time  1.2665576934814453
Epoch  300  time  1.3348362445831299
Epoch  300  loss  0.3210856443928439 correct 50
Epoch  301  time  1.27272367477417
Epoch  302  time  1.8306124210357666
Epoch  303  time  1.4379799365997314
Epoch  304  time  1.2508866786956787
Epoch  305  time  1.309248685836792
Epoch  306  time  1.258542776107788
Epoch  307  time  1.2496297359466553
Epoch  308  time  1.2492411136627197
Epoch  309  time  1.3092117309570312
Epoch  310  time  1.247190237045288
Epoch  310  loss  0.2055371212571514 correct 50
Epoch  311  time  1.8174176216125488
Epoch  312  time  1.5101654529571533
Epoch  313  time  1.2461893558502197
Epoch  314  time  1.3509557247161865
Epoch  315  time  1.2432374954223633
Epoch  316  time  1.2459330558776855
Epoch  317  time  1.2450897693634033
Epoch  318  time  1.2614686489105225
Epoch  319  time  1.3045461177825928
Epoch  320  time  1.670607566833496
Epoch  320  loss  0.0007356229080873201 correct 50
Epoch  321  time  1.5671391487121582
Epoch  322  time  1.247633934020996
Epoch  323  time  1.2522761821746826
Epoch  324  time  1.3135418891906738
Epoch  325  time  1.2593059539794922
Epoch  326  time  1.2499592304229736
Epoch  327  time  1.256739616394043
Epoch  328  time  1.2578237056732178
Epoch  329  time  1.6548898220062256
Epoch  330  time  1.7210659980773926
Epoch  330  loss  0.08038391346289574 correct 50
Epoch  331  time  1.2477104663848877
Epoch  332  time  1.2596831321716309
Epoch  333  time  1.253732681274414
Epoch  334  time  1.2484982013702393
Epoch  335  time  1.3087782859802246
Epoch  336  time  1.2769031524658203
Epoch  337  time  1.2825102806091309
Epoch  338  time  1.6163311004638672
Epoch  339  time  1.7093908786773682
Epoch  340  time  1.304814338684082
Epoch  340  loss  0.0759544521927877 correct 50
Epoch  341  time  1.2500901222229004
Epoch  342  time  1.2504065036773682
Epoch  343  time  1.256272315979004
Epoch  344  time  1.2670080661773682
Epoch  345  time  1.29710054397583
Epoch  346  time  1.256136178970337
Epoch  347  time  1.5441431999206543
Epoch  348  time  1.7486293315887451
Epoch  349  time  1.251420021057129
Epoch  350  time  1.2943651676177979
Epoch  350  loss  0.00022148344253406697 correct 50
Epoch  351  time  1.2553794384002686
Epoch  352  time  1.2628874778747559
Epoch  353  time  1.2510590553283691
Epoch  354  time  1.2552523612976074
Epoch  355  time  1.8556835651397705
Epoch  356  time  2.0376524925231934
Epoch  357  time  1.6061744689941406
Epoch  358  time  1.2961819171905518
Epoch  359  time  1.2463490962982178
Epoch  360  time  1.3024065494537354
Epoch  360  loss  0.11896708368821243 correct 50
Epoch  361  time  1.2455193996429443
Epoch  362  time  1.259929895401001
Epoch  363  time  1.2435200214385986
Epoch  364  time  1.2449395656585693
Epoch  365  time  1.7246730327606201
Epoch  366  time  1.6509616374969482
Epoch  367  time  1.24283766746521
Epoch  368  time  1.2477343082427979
Epoch  369  time  1.2556161880493164
Epoch  370  time  1.2992660999298096
Epoch  370  loss  0.002392108909623735 correct 50
Epoch  371  time  1.2451846599578857
Epoch  372  time  1.2509863376617432
Epoch  373  time  1.256465196609497
Epoch  374  time  1.5264983177185059
Epoch  375  time  1.7911090850830078
Epoch  376  time  1.245389699935913
Epoch  377  time  1.2463946342468262
Epoch  378  time  1.2529280185699463
Epoch  379  time  1.2467052936553955
Epoch  380  time  1.336402416229248
Epoch  380  loss  0.0012057391366560282 correct 50
Epoch  381  time  1.2488396167755127
Epoch  382  time  1.2532000541687012
Epoch  383  time  1.4828712940216064
Epoch  384  time  1.809514045715332
Epoch  385  time  1.2983815670013428
Epoch  386  time  1.246504306793213
Epoch  387  time  1.2467877864837646
Epoch  388  time  1.273205041885376
Epoch  389  time  1.2638542652130127
Epoch  390  time  1.3071305751800537
Epoch  390  loss  0.18156209427542397 correct 50
Epoch  391  time  1.2711875438690186
Epoch  392  time  1.4643938541412354
Epoch  393  time  1.840597152709961
Epoch  394  time  1.2556958198547363
Epoch  395  time  1.3137736320495605
Epoch  396  time  1.2461721897125244
Epoch  397  time  1.2540624141693115
Epoch  398  time  1.2789726257324219
Epoch  399  time  1.2436258792877197
Epoch  400  time  1.2418463230133057
Epoch  400  loss  0.07777277484525612 correct 50
Epoch  401  time  1.4704530239105225
Epoch  402  time  1.9258172512054443
Epoch  403  time  1.2474441528320312
Epoch  404  time  1.2452888488769531
Epoch  405  time  1.2522296905517578
Epoch  406  time  1.3049964904785156
Epoch  407  time  1.238555908203125
Epoch  408  time  1.2425804138183594
Epoch  409  time  1.2451813220977783
Epoch  410  time  1.2584965229034424
Epoch  410  loss  0.028470039270625167 correct 50
Epoch  411  time  1.9191267490386963
Epoch  412  time  1.3643929958343506
Epoch  413  time  1.247025489807129
Epoch  414  time  1.2450435161590576
Epoch  415  time  1.2718737125396729
Epoch  416  time  1.3111224174499512
Epoch  417  time  1.2539472579956055
Epoch  418  time  1.2620079517364502
Epoch  419  time  1.2466371059417725
Epoch  420  time  1.8021478652954102
Epoch  420  loss  0.2063489078523702 correct 50
Epoch  421  time  1.4991271495819092
Epoch  422  time  1.2599716186523438
Epoch  423  time  1.2764379978179932
Epoch  424  time  1.2456698417663574
Epoch  425  time  1.2868263721466064
Epoch  426  time  1.3068811893463135
Epoch  427  time  1.24515962600708
Epoch  428  time  1.2453536987304688
Epoch  429  time  1.7558300495147705
Epoch  430  time  1.5209324359893799
Epoch  430  loss  0.01701900757954156 correct 50
Epoch  431  time  1.3205139636993408
Epoch  432  time  1.2566149234771729
Epoch  433  time  1.2528705596923828
Epoch  434  time  1.2588579654693604
Epoch  435  time  1.2609643936157227
Epoch  436  time  1.295961856842041
Epoch  437  time  1.2589819431304932
Epoch  438  time  1.7075581550598145
Epoch  439  time  1.6039316654205322
Epoch  440  time  1.257399320602417
Epoch  440  loss  0.08818773213276167 correct 50
Epoch  441  time  1.311065435409546
Epoch  442  time  1.2602250576019287
Epoch  443  time  1.2390358448028564
Epoch  444  time  1.2408955097198486
Epoch  445  time  1.2510967254638672
Epoch  446  time  1.2991926670074463
Epoch  447  time  1.634423017501831
Epoch  448  time  1.6700365543365479
Epoch  449  time  1.2480084896087646
Epoch  450  time  1.2455668449401855
Epoch  450  loss  0.04164887082543214 correct 50
Epoch  451  time  1.3098478317260742
Epoch  452  time  1.2489867210388184
Epoch  453  time  1.249889850616455
Epoch  454  time  1.2408208847045898
Epoch  455  time  1.2577476501464844
Epoch  456  time  1.6203656196594238
Epoch  457  time  1.7442994117736816
Epoch  458  time  1.2575552463531494
Epoch  459  time  1.254699468612671
Epoch  460  time  1.2541208267211914
Epoch  460  loss  0.16779861785111208 correct 50
Epoch  461  time  1.30946683883667
Epoch  462  time  1.2551944255828857
Epoch  463  time  1.2477953433990479
Epoch  464  time  1.267179012298584
Epoch  465  time  1.4684422016143799
Epoch  466  time  1.8506309986114502
Epoch  467  time  1.245718002319336
Epoch  468  time  1.2430763244628906
Epoch  469  time  1.2724189758300781
Epoch  470  time  1.2583346366882324
Epoch  470  loss  0.0022671487132390317 correct 50
Epoch  471  time  1.3129634857177734
Epoch  472  time  1.2633662223815918
Epoch  473  time  1.2498221397399902
Epoch  474  time  1.4222650527954102
Epoch  475  time  1.8543603420257568
Epoch  476  time  1.3065791130065918
Epoch  477  time  1.2435309886932373
Epoch  478  time  1.2540102005004883
Epoch  479  time  1.2762773036956787
Epoch  480  time  1.244093894958496
Epoch  480  loss  0.01235013897887169 correct 50
Epoch  481  time  1.342550277709961
Epoch  482  time  1.2395906448364258
Epoch  483  time  1.376312017440796
Epoch  484  time  1.8570590019226074
Epoch  485  time  1.330103874206543
Epoch  486  time  1.2699341773986816
Epoch  487  time  1.303091287612915
Epoch  488  time  1.247941493988037
Epoch  489  time  1.2513301372528076
Epoch  490  time  1.2403645515441895
Epoch  490  loss  0.07561466871589193 correct 50
Epoch  491  time  1.2655525207519531
Epoch  492  time  1.3418381214141846
Epoch  493  time  1.867497205734253
Epoch  494  time  1.4002339839935303
Epoch  495  time  1.2594027519226074
Epoch  496  time  1.2460505962371826
Epoch  497  time  1.2970070838928223
Epoch  498  time  1.262770175933838
Epoch  499  time  1.2419812679290771
```
</details>

python run_fast_tensor.py --BACKEND cpu --HIDDEN 100 --DATASET simple --RATE 0.05

*Average time per epoch 0.1852697558403015 (for 500 epochs)*

<details>
<summary>View log</summary>
Epoch  0  time  5.887956857681274
Epoch  0  loss  7.535361536952158 correct 33
Epoch  1  time  0.16381430625915527
Epoch  2  time  0.16514277458190918
Epoch  3  time  0.1948108673095703
Epoch  4  time  0.17824983596801758
Epoch  5  time  0.1838667392730713
Epoch  6  time  0.20431280136108398
Epoch  7  time  0.25571417808532715
Epoch  8  time  0.1748208999633789
Epoch  9  time  0.20282697677612305
Epoch  10  time  0.1915569305419922
Epoch  10  loss  2.345161539633469 correct 48
Epoch  11  time  0.19360709190368652
Epoch  12  time  0.1653439998626709
Epoch  13  time  0.17265987396240234
Epoch  14  time  0.16782903671264648
Epoch  15  time  0.17743778228759766
Epoch  16  time  0.2325141429901123
Epoch  17  time  0.22083592414855957
Epoch  18  time  0.19356203079223633
Epoch  19  time  0.17486906051635742
Epoch  20  time  0.20766115188598633
Epoch  20  loss  2.3863617310189524 correct 46
Epoch  21  time  0.18579888343811035
Epoch  22  time  0.21953105926513672
Epoch  23  time  0.18708395957946777
Epoch  24  time  0.2013719081878662
Epoch  25  time  0.21823883056640625
Epoch  26  time  0.19688105583190918
Epoch  27  time  0.17345809936523438
Epoch  28  time  0.2048499584197998
Epoch  29  time  0.18564105033874512
Epoch  30  time  0.20253419876098633
Epoch  30  loss  0.5708908307055095 correct 49
Epoch  31  time  0.1877140998840332
Epoch  32  time  0.18019580841064453
Epoch  33  time  0.20271921157836914
Epoch  34  time  0.17176413536071777
Epoch  35  time  0.20018720626831055
Epoch  36  time  0.17844104766845703
Epoch  37  time  0.20282483100891113
Epoch  38  time  0.1849071979522705
Epoch  39  time  0.2368471622467041
Epoch  40  time  0.18142127990722656
Epoch  40  loss  1.8872271072233888 correct 50
Epoch  41  time  0.19404983520507812
Epoch  42  time  0.18351221084594727
Epoch  43  time  0.1759181022644043
Epoch  44  time  0.1999361515045166
Epoch  45  time  0.18340110778808594
Epoch  46  time  0.20365476608276367
Epoch  47  time  0.19306612014770508
Epoch  48  time  0.20757317543029785
Epoch  49  time  0.19589805603027344
Epoch  50  time  0.1943826675415039
Epoch  50  loss  0.7552894978989789 correct 50
Epoch  51  time  0.19062495231628418
Epoch  52  time  0.19229602813720703
Epoch  53  time  0.2901449203491211
Epoch  54  time  0.17684292793273926
Epoch  55  time  0.16718196868896484
Epoch  56  time  0.17297601699829102
Epoch  57  time  0.15769028663635254
Epoch  58  time  0.1741650104522705
Epoch  59  time  0.17974400520324707
Epoch  60  time  0.1781930923461914
Epoch  60  loss  0.9113574881569695 correct 49
Epoch  61  time  0.1812429428100586
Epoch  62  time  0.18991684913635254
Epoch  63  time  0.17159008979797363
Epoch  64  time  0.18356800079345703
Epoch  65  time  0.1829679012298584
Epoch  66  time  0.18436121940612793
Epoch  67  time  0.17256712913513184
Epoch  68  time  0.17095398902893066
Epoch  69  time  0.17464804649353027
Epoch  70  time  0.17078804969787598
Epoch  70  loss  1.5228799093990335 correct 50
Epoch  71  time  0.16740679740905762
Epoch  72  time  0.16883492469787598
Epoch  73  time  0.20383095741271973
Epoch  74  time  0.2077479362487793
Epoch  75  time  0.21126079559326172
Epoch  76  time  0.259411096572876
Epoch  77  time  0.17149996757507324
Epoch  78  time  0.17443084716796875
Epoch  79  time  0.19904398918151855
Epoch  80  time  0.2009599208831787
Epoch  80  loss  0.37211754005537384 correct 49
Epoch  81  time  0.1940310001373291
Epoch  82  time  0.16550397872924805
Epoch  83  time  0.16338706016540527
Epoch  84  time  0.2028651237487793
Epoch  85  time  0.18570303916931152
Epoch  86  time  0.1703810691833496
Epoch  87  time  0.1637730598449707
Epoch  88  time  0.1751561164855957
Epoch  89  time  0.1783921718597412
Epoch  90  time  0.17317581176757812
Epoch  90  loss  1.1516587073804334 correct 50
Epoch  91  time  0.18728113174438477
Epoch  92  time  0.17987895011901855
Epoch  93  time  0.18266582489013672
Epoch  94  time  0.18225979804992676
Epoch  95  time  0.1859898567199707
Epoch  96  time  0.17198610305786133
Epoch  97  time  0.2008042335510254
Epoch  98  time  0.1826012134552002
Epoch  99  time  0.18101191520690918
Epoch  100  time  0.21282196044921875
Epoch  100  loss  0.9267819124599955 correct 50
Epoch  101  time  0.17291593551635742
Epoch  102  time  0.16646695137023926
Epoch  103  time  0.17104005813598633
Epoch  104  time  0.1734468936920166
Epoch  105  time  0.16317391395568848
Epoch  106  time  0.1629619598388672
Epoch  107  time  0.21133732795715332
Epoch  108  time  0.18289780616760254
Epoch  109  time  0.16827893257141113
Epoch  110  time  0.17256402969360352
Epoch  110  loss  0.5479532132140473 correct 49
Epoch  111  time  0.18205022811889648
Epoch  112  time  0.19022107124328613
Epoch  113  time  0.17691922187805176
Epoch  114  time  0.17025208473205566
Epoch  115  time  0.16283392906188965
Epoch  116  time  0.16337800025939941
Epoch  117  time  0.16436100006103516
Epoch  118  time  0.16994261741638184
Epoch  119  time  0.1660008430480957
Epoch  120  time  0.16526579856872559
Epoch  120  loss  0.23977412826544645 correct 49
Epoch  121  time  0.1635432243347168
Epoch  122  time  0.16382479667663574
Epoch  123  time  0.1725902557373047
Epoch  124  time  0.16339492797851562
Epoch  125  time  0.16342806816101074
Epoch  126  time  0.16535615921020508
Epoch  127  time  0.1610410213470459
Epoch  128  time  0.16469812393188477
Epoch  129  time  0.17585110664367676
Epoch  130  time  0.17760682106018066
Epoch  130  loss  0.7292280182758832 correct 50
Epoch  131  time  0.1786329746246338
Epoch  132  time  0.16643190383911133
Epoch  133  time  0.16556692123413086
Epoch  134  time  0.16391921043395996
Epoch  135  time  0.1789388656616211
Epoch  136  time  0.17459511756896973
Epoch  137  time  0.17249202728271484
Epoch  138  time  0.1772768497467041
Epoch  139  time  0.16393804550170898
Epoch  140  time  0.17537307739257812
Epoch  140  loss  0.049329005958591736 correct 50
Epoch  141  time  0.17113804817199707
Epoch  142  time  0.16364002227783203
Epoch  143  time  0.1708991527557373
Epoch  144  time  0.17218375205993652
Epoch  145  time  0.16456127166748047
Epoch  146  time  0.17338109016418457
Epoch  147  time  0.1711418628692627
Epoch  148  time  0.17319297790527344
Epoch  149  time  0.17378878593444824
Epoch  150  time  0.23144006729125977
Epoch  150  loss  0.010215444757708145 correct 49
Epoch  151  time  0.17717313766479492
Epoch  152  time  0.17659687995910645
Epoch  153  time  0.17834877967834473
Epoch  154  time  0.17241382598876953
Epoch  155  time  0.17482209205627441
Epoch  156  time  0.16984868049621582
Epoch  157  time  0.1970980167388916
Epoch  158  time  0.18506908416748047
Epoch  159  time  0.18190598487854004
Epoch  160  time  0.17536306381225586
Epoch  160  loss  0.21895481605275552 correct 49
Epoch  161  time  0.17412090301513672
Epoch  162  time  0.17079806327819824
Epoch  163  time  0.1712329387664795
Epoch  164  time  0.17638707160949707
Epoch  165  time  0.169907808303833
Epoch  166  time  0.21776604652404785
Epoch  167  time  0.18043208122253418
Epoch  168  time  0.17037296295166016
Epoch  169  time  0.17052888870239258
Epoch  170  time  0.17285418510437012
Epoch  170  loss  0.953587281622586 correct 50
Epoch  171  time  0.1720571517944336
Epoch  172  time  0.17275691032409668
Epoch  173  time  0.17973089218139648
Epoch  174  time  0.17353606224060059
Epoch  175  time  0.1800689697265625
Epoch  176  time  0.21772980690002441
Epoch  177  time  0.1855940818786621
Epoch  178  time  0.19257712364196777
Epoch  179  time  0.1680448055267334
Epoch  180  time  0.2074129581451416
Epoch  180  loss  1.0780484587121013 correct 50
Epoch  181  time  0.18383097648620605
Epoch  182  time  0.20143985748291016
Epoch  183  time  0.2003629207611084
Epoch  184  time  0.18392610549926758
Epoch  185  time  0.1722559928894043
Epoch  186  time  0.180067777633667
Epoch  187  time  0.16733407974243164
Epoch  188  time  0.16068792343139648
Epoch  189  time  0.16571903228759766
Epoch  190  time  0.18859076499938965
Epoch  190  loss  0.27732884303473654 correct 50
Epoch  191  time  0.18601107597351074
Epoch  192  time  0.17064809799194336
Epoch  193  time  0.17789506912231445
Epoch  194  time  0.17002511024475098
Epoch  195  time  0.1710832118988037
Epoch  196  time  0.16374564170837402
Epoch  197  time  0.17734408378601074
Epoch  198  time  0.21221590042114258
Epoch  199  time  0.17731189727783203
Epoch  200  time  0.16170072555541992
Epoch  200  loss  0.013785118556386564 correct 50
Epoch  201  time  0.16932201385498047
Epoch  202  time  0.1681218147277832
Epoch  203  time  0.1691420078277588
Epoch  204  time  0.17321085929870605
Epoch  205  time  0.18042707443237305
Epoch  206  time  0.1810932159423828
Epoch  207  time  0.1727440357208252
Epoch  208  time  0.18611502647399902
Epoch  209  time  0.17604613304138184
Epoch  210  time  0.2590813636779785
Epoch  210  loss  0.058681796510827056 correct 49
Epoch  211  time  0.1705782413482666
Epoch  212  time  0.1739368438720703
Epoch  213  time  0.23157119750976562
Epoch  214  time  0.17859387397766113
Epoch  215  time  0.2271432876586914
Epoch  216  time  0.18376922607421875
Epoch  217  time  0.17542076110839844
Epoch  218  time  0.16251897811889648
Epoch  219  time  0.1633760929107666
Epoch  220  time  0.16189002990722656
Epoch  220  loss  0.34590365307879894 correct 50
Epoch  221  time  0.1705331802368164
Epoch  222  time  0.1612088680267334
Epoch  223  time  0.16588068008422852
Epoch  224  time  0.1657118797302246
Epoch  225  time  0.16471195220947266
Epoch  226  time  0.16564297676086426
Epoch  227  time  0.16070890426635742
Epoch  228  time  0.15990018844604492
Epoch  229  time  0.16641998291015625
Epoch  230  time  0.16925811767578125
Epoch  230  loss  0.21599912055684703 correct 50
Epoch  231  time  0.1734931468963623
Epoch  232  time  0.17003703117370605
Epoch  233  time  0.16366004943847656
Epoch  234  time  0.16482925415039062
Epoch  235  time  0.16475391387939453
Epoch  236  time  0.17574620246887207
Epoch  237  time  0.16684317588806152
Epoch  238  time  0.16684794425964355
Epoch  239  time  0.16410326957702637
Epoch  240  time  0.17944788932800293
Epoch  240  loss  0.0001464649415064014 correct 50
Epoch  241  time  0.16103219985961914
Epoch  242  time  0.1673870086669922
Epoch  243  time  0.17162704467773438
Epoch  244  time  0.1675410270690918
Epoch  245  time  0.16675186157226562
Epoch  246  time  0.164290189743042
Epoch  247  time  0.15896010398864746
Epoch  248  time  0.16278815269470215
Epoch  249  time  0.17258405685424805
Epoch  250  time  0.17173409461975098
Epoch  250  loss  0.04595156940483051 correct 49
Epoch  251  time  0.17542600631713867
Epoch  252  time  0.16673684120178223
Epoch  253  time  0.1705179214477539
Epoch  254  time  0.16099214553833008
Epoch  255  time  0.16568922996520996
Epoch  256  time  0.1640620231628418
Epoch  257  time  0.16446995735168457
Epoch  258  time  0.16509318351745605
Epoch  259  time  0.16903400421142578
Epoch  260  time  0.16501808166503906
Epoch  260  loss  0.20023160023412057 correct 50
Epoch  261  time  0.16180896759033203
Epoch  262  time  0.16449999809265137
Epoch  263  time  0.17978119850158691
Epoch  264  time  0.15672922134399414
Epoch  265  time  0.16498804092407227
Epoch  266  time  0.16715598106384277
Epoch  267  time  0.16506695747375488
Epoch  268  time  0.17579984664916992
Epoch  269  time  0.17119598388671875
Epoch  270  time  0.17216086387634277
Epoch  270  loss  0.42576007795994525 correct 50
Epoch  271  time  0.1679859161376953
Epoch  272  time  0.1631460189819336
Epoch  273  time  0.1654071807861328
Epoch  274  time  0.2028028964996338
Epoch  275  time  0.18394017219543457
Epoch  276  time  0.1681232452392578
Epoch  277  time  0.16220784187316895
Epoch  278  time  0.16233015060424805
Epoch  279  time  0.168503999710083
Epoch  280  time  0.168410062789917
Epoch  280  loss  0.1595672778375988 correct 49
Epoch  281  time  0.16130805015563965
Epoch  282  time  0.16973185539245605
Epoch  283  time  0.1622631549835205
Epoch  284  time  0.16205787658691406
Epoch  285  time  0.16265392303466797
Epoch  286  time  0.16367411613464355
Epoch  287  time  0.16150903701782227
Epoch  288  time  0.16587591171264648
Epoch  289  time  0.16933894157409668
Epoch  290  time  0.16290783882141113
Epoch  290  loss  0.192700073364868 correct 50
Epoch  291  time  0.17473816871643066
Epoch  292  time  0.15795397758483887
Epoch  293  time  0.16116690635681152
Epoch  294  time  0.1655750274658203
Epoch  295  time  0.16139888763427734
Epoch  296  time  0.17112302780151367
Epoch  297  time  0.1718580722808838
Epoch  298  time  0.1619739532470703
Epoch  299  time  0.17441296577453613
Epoch  300  time  0.16825032234191895
Epoch  300  loss  0.28556030516291564 correct 50
Epoch  301  time  0.16465520858764648
Epoch  302  time  0.16591787338256836
Epoch  303  time  0.1672971248626709
Epoch  304  time  0.16403794288635254
Epoch  305  time  0.16641783714294434
Epoch  306  time  0.16481781005859375
Epoch  307  time  0.16626906394958496
Epoch  308  time  0.1627497673034668
Epoch  309  time  0.16675901412963867
Epoch  310  time  0.16507291793823242
Epoch  310  loss  0.641378122540491 correct 50
Epoch  311  time  0.1681210994720459
Epoch  312  time  0.16638803482055664
Epoch  313  time  0.16643190383911133
Epoch  314  time  0.1585068702697754
Epoch  315  time  0.15865492820739746
Epoch  316  time  0.168168306350708
Epoch  317  time  0.16724824905395508
Epoch  318  time  0.16327404975891113
Epoch  319  time  0.1688828468322754
Epoch  320  time  0.15886306762695312
Epoch  320  loss  0.1609508818138793 correct 50
Epoch  321  time  0.16565203666687012
Epoch  322  time  0.16640901565551758
Epoch  323  time  0.16145920753479004
Epoch  324  time  0.16632604598999023
Epoch  325  time  0.17020320892333984
Epoch  326  time  0.1646559238433838
Epoch  327  time  0.16532301902770996
Epoch  328  time  0.16695094108581543
Epoch  329  time  0.18331694602966309
Epoch  330  time  0.16355609893798828
Epoch  330  loss  0.1575352899905957 correct 50
Epoch  331  time  0.1672499179840088
Epoch  332  time  0.1681830883026123
Epoch  333  time  0.1671738624572754
Epoch  334  time  0.16958284378051758
Epoch  335  time  0.16019272804260254
Epoch  336  time  0.16426682472229004
Epoch  337  time  0.16872501373291016
Epoch  338  time  0.16435813903808594
Epoch  339  time  0.16971302032470703
Epoch  340  time  0.16621899604797363
Epoch  340  loss  1.9851220031154277e-05 correct 50
Epoch  341  time  0.1633291244506836
Epoch  342  time  0.16961383819580078
Epoch  343  time  0.15862083435058594
Epoch  344  time  0.16417694091796875
Epoch  345  time  0.1692368984222412
Epoch  346  time  0.15989899635314941
Epoch  347  time  0.16571426391601562
Epoch  348  time  0.16870403289794922
Epoch  349  time  0.16360902786254883
Epoch  350  time  0.16815423965454102
Epoch  350  loss  0.04197903825625355 correct 50
Epoch  351  time  0.17258906364440918
Epoch  352  time  0.1692218780517578
Epoch  353  time  0.17098093032836914
Epoch  354  time  0.19053077697753906
Epoch  355  time  0.1836540699005127
Epoch  356  time  0.16987919807434082
Epoch  357  time  0.16739320755004883
Epoch  358  time  0.17792677879333496
Epoch  359  time  0.16577911376953125
Epoch  360  time  0.16513895988464355
Epoch  360  loss  0.4774696048607814 correct 50
Epoch  361  time  0.17053484916687012
Epoch  362  time  0.16978883743286133
Epoch  363  time  0.16592884063720703
Epoch  364  time  0.16770505905151367
Epoch  365  time  0.1714339256286621
Epoch  366  time  0.16194987297058105
Epoch  367  time  0.1663200855255127
Epoch  368  time  0.16774511337280273
Epoch  369  time  0.16494011878967285
Epoch  370  time  0.16079092025756836
Epoch  370  loss  0.39988879678904266 correct 50
Epoch  371  time  0.16042113304138184
Epoch  372  time  0.16908025741577148
Epoch  373  time  0.17624592781066895
Epoch  374  time  0.17240190505981445
Epoch  375  time  0.16194605827331543
Epoch  376  time  0.16063308715820312
Epoch  377  time  0.16736984252929688
Epoch  378  time  0.16686201095581055
Epoch  379  time  0.16298198699951172
Epoch  380  time  0.16341280937194824
Epoch  380  loss  0.003096086786334436 correct 50
Epoch  381  time  0.16132903099060059
Epoch  382  time  0.16283488273620605
Epoch  383  time  0.16086983680725098
Epoch  384  time  0.16869521141052246
Epoch  385  time  0.16463088989257812
Epoch  386  time  0.1626911163330078
Epoch  387  time  0.16733431816101074
Epoch  388  time  0.18200206756591797
Epoch  389  time  0.17324185371398926
Epoch  390  time  0.17420697212219238
Epoch  390  loss  0.5211072964543042 correct 50
Epoch  391  time  0.16575312614440918
Epoch  392  time  0.16749000549316406
Epoch  393  time  0.16287779808044434
Epoch  394  time  0.17314791679382324
Epoch  395  time  0.16707205772399902
Epoch  396  time  0.16074419021606445
Epoch  397  time  0.1609950065612793
Epoch  398  time  0.16819190979003906
Epoch  399  time  0.16617918014526367
Epoch  400  time  0.15924620628356934
Epoch  400  loss  0.3263104200681983 correct 50
Epoch  401  time  0.16977620124816895
Epoch  402  time  0.16516709327697754
Epoch  403  time  0.1666431427001953
Epoch  404  time  0.16538596153259277
Epoch  405  time  0.16504192352294922
Epoch  406  time  0.16978096961975098
Epoch  407  time  0.1618821620941162
Epoch  408  time  0.16629886627197266
Epoch  409  time  0.16660714149475098
Epoch  410  time  0.16779708862304688
Epoch  410  loss  0.0014021299614830447 correct 50
Epoch  411  time  0.16889095306396484
Epoch  412  time  0.16628098487854004
Epoch  413  time  0.1674339771270752
Epoch  414  time  0.1683199405670166
Epoch  415  time  0.1668698787689209
Epoch  416  time  0.1676650047302246
Epoch  417  time  0.18563079833984375
Epoch  418  time  0.1676959991455078
Epoch  419  time  0.16585087776184082
Epoch  420  time  0.16673994064331055
Epoch  420  loss  0.07325035922145327 correct 50
Epoch  421  time  0.16779184341430664
Epoch  422  time  0.16463613510131836
Epoch  423  time  0.1660597324371338
Epoch  424  time  0.16379690170288086
Epoch  425  time  0.16440320014953613
Epoch  426  time  0.16626310348510742
Epoch  427  time  0.1679229736328125
Epoch  428  time  0.16202282905578613
Epoch  429  time  0.16398310661315918
Epoch  430  time  0.16109895706176758
Epoch  430  loss  0.6373073322167856 correct 49
Epoch  431  time  0.17552900314331055
Epoch  432  time  0.16319894790649414
Epoch  433  time  0.16134309768676758
Epoch  434  time  0.16558122634887695
Epoch  435  time  0.16393303871154785
Epoch  436  time  0.1698472499847412
Epoch  437  time  0.17371606826782227
Epoch  438  time  0.16630911827087402
Epoch  439  time  0.1724100112915039
Epoch  440  time  0.1634807586669922
Epoch  440  loss  0.02600625659123658 correct 50
Epoch  441  time  0.16798710823059082
Epoch  442  time  0.16652488708496094
Epoch  443  time  0.17213892936706543
Epoch  444  time  0.1608750820159912
Epoch  445  time  0.16903090476989746
Epoch  446  time  0.16777491569519043
Epoch  447  time  0.17656207084655762
Epoch  448  time  0.16839599609375
Epoch  449  time  0.16875410079956055
Epoch  450  time  0.16787385940551758
Epoch  450  loss  0.6494501422244477 correct 50
Epoch  451  time  0.16405296325683594
Epoch  452  time  0.17165398597717285
Epoch  453  time  0.17258691787719727
Epoch  454  time  0.16506385803222656
Epoch  455  time  0.16956186294555664
Epoch  456  time  0.17940306663513184
Epoch  457  time  0.17120814323425293
Epoch  458  time  0.15877723693847656
Epoch  459  time  0.1686108112335205
Epoch  460  time  0.16599607467651367
Epoch  460  loss  0.7029561711265986 correct 50
Epoch  461  time  0.15939903259277344
Epoch  462  time  0.16048502922058105
Epoch  463  time  0.16126394271850586
Epoch  464  time  0.1684589385986328
Epoch  465  time  0.1695091724395752
Epoch  466  time  0.16179299354553223
Epoch  467  time  0.1661698818206787
Epoch  468  time  0.16578197479248047
Epoch  469  time  0.15597105026245117
Epoch  470  time  0.16271376609802246
Epoch  470  loss  0.06849321787133145 correct 50
Epoch  471  time  0.1639871597290039
Epoch  472  time  0.15601801872253418
Epoch  473  time  0.1625080108642578
Epoch  474  time  0.16704678535461426
Epoch  475  time  0.1640160083770752
Epoch  476  time  0.16501712799072266
Epoch  477  time  0.17401671409606934
Epoch  478  time  0.16466784477233887
Epoch  479  time  0.1612682342529297
Epoch  480  time  0.17027711868286133
Epoch  480  loss  0.04333463485053022 correct 50
Epoch  481  time  0.16582393646240234
Epoch  482  time  0.17294907569885254
Epoch  483  time  0.16259407997131348
Epoch  484  time  0.1630079746246338
Epoch  485  time  0.16833114624023438
Epoch  486  time  0.1637439727783203
Epoch  487  time  0.16639423370361328
Epoch  488  time  0.16684412956237793
Epoch  489  time  0.1669297218322754
Epoch  490  time  0.1639690399169922
Epoch  490  loss  0.04461924193815575 correct 50
Epoch  491  time  0.16284990310668945
Epoch  492  time  0.16496896743774414
Epoch  493  time  0.16489601135253906
Epoch  494  time  0.16853094100952148
Epoch  495  time  0.16298627853393555
Epoch  496  time  0.16672086715698242
Epoch  497  time  0.1642160415649414
Epoch  498  time  0.1683361530303955
Epoch  499  time  0.1961989402770996
</details>

## XOR Dataset

python3.12 project/run_fast_tensor.py --BACKEND gpu --HIDDEN 100 --DATASET xor --RATE 0.05

*Average time per epoch 1.356607183933258 (for 500 epochs)*

<details>
<summary>View log</summary>
```
Epoch  0  loss  6.887350291036325 correct 41
Epoch  1  time  1.2633967399597168
Epoch  2  time  1.312666893005371
Epoch  3  time  1.2480316162109375
Epoch  4  time  1.247997522354126
Epoch  5  time  1.2547760009765625
Epoch  6  time  1.2809717655181885
Epoch  7  time  1.3127787113189697
Epoch  8  time  1.4459214210510254
Epoch  9  time  1.8414108753204346
Epoch  10  time  1.2451481819152832
Epoch  10  loss  4.6716637249578445 correct 41
Epoch  11  time  1.2737343311309814
Epoch  12  time  1.3078818321228027
Epoch  13  time  1.3063385486602783
Epoch  14  time  1.274932622909546
Epoch  15  time  1.2589530944824219
Epoch  16  time  1.2538740634918213
Epoch  17  time  1.4979307651519775
Epoch  18  time  1.910247802734375
Epoch  19  time  1.2507975101470947
Epoch  20  time  1.249497413635254
Epoch  20  loss  5.209442609032938 correct 48
Epoch  21  time  1.2653923034667969
Epoch  22  time  1.3039231300354004
Epoch  23  time  1.245069980621338
Epoch  24  time  1.258432388305664
Epoch  25  time  1.2695777416229248
Epoch  26  time  1.382333517074585
Epoch  27  time  1.9287278652191162
Epoch  28  time  1.3062222003936768
Epoch  29  time  1.259772539138794
Epoch  30  time  1.2565829753875732
Epoch  30  loss  4.7603310186667676 correct 45
Epoch  31  time  1.2553870677947998
Epoch  32  time  1.298079013824463
Epoch  33  time  1.2515983581542969
Epoch  34  time  1.2565100193023682
Epoch  35  time  1.2947816848754883
Epoch  36  time  1.83778977394104
Epoch  37  time  1.4713306427001953
Epoch  38  time  1.260730504989624
Epoch  39  time  1.2721259593963623
Epoch  40  time  1.270125389099121
Epoch  40  loss  2.5163021162889256 correct 47
Epoch  41  time  1.2440919876098633
Epoch  42  time  1.3151960372924805
Epoch  43  time  1.2636301517486572
Epoch  44  time  1.2522737979888916
Epoch  45  time  1.8594915866851807
Epoch  46  time  1.4531562328338623
Epoch  47  time  1.3147189617156982
Epoch  48  time  1.2503209114074707
Epoch  49  time  1.2627265453338623
Epoch  50  time  1.2805967330932617
Epoch  50  loss  4.2292410725750935 correct 47
Epoch  51  time  1.2444849014282227
Epoch  52  time  1.296614646911621
Epoch  53  time  1.2563374042510986
Epoch  54  time  1.7999649047851562
Epoch  55  time  1.5117156505584717
Epoch  56  time  1.2501027584075928
Epoch  57  time  1.2944831848144531
Epoch  58  time  1.2744286060333252
Epoch  59  time  1.2474234104156494
Epoch  60  time  1.249765396118164
Epoch  60  loss  3.020708957815045 correct 48
Epoch  61  time  1.2542815208435059
Epoch  62  time  1.3156521320343018
Epoch  63  time  1.7722153663635254
Epoch  64  time  1.5535962581634521
Epoch  65  time  1.2503094673156738
Epoch  66  time  1.279784917831421
Epoch  67  time  1.2666490077972412
Epoch  68  time  1.326697826385498
Epoch  69  time  1.2414352893829346
Epoch  70  time  1.2442119121551514
Epoch  70  loss  1.6672286585951783 correct 49
Epoch  71  time  1.2464659214019775
Epoch  72  time  1.6996135711669922
Epoch  73  time  1.6651818752288818
Epoch  74  time  1.2484252452850342
Epoch  75  time  1.2412745952606201
Epoch  76  time  1.2471084594726562
Epoch  77  time  1.2516636848449707
Epoch  78  time  1.295309066772461
Epoch  79  time  1.2415907382965088
Epoch  80  time  1.2448513507843018
Epoch  80  loss  4.026717190337825 correct 47
Epoch  81  time  1.6205952167510986
Epoch  82  time  1.6787686347961426
Epoch  83  time  1.3329434394836426
Epoch  84  time  1.2429466247558594
Epoch  85  time  1.2761952877044678
Epoch  86  time  1.2516155242919922
Epoch  87  time  1.253485918045044
Epoch  88  time  1.3172063827514648
Epoch  89  time  1.2583985328674316
Epoch  90  time  1.566838264465332
Epoch  90  loss  3.778666724907266 correct 47
Epoch  91  time  1.7054996490478516
Epoch  92  time  1.2632100582122803
Epoch  93  time  1.3041651248931885
Epoch  94  time  1.2647438049316406
Epoch  95  time  1.275423288345337
Epoch  96  time  1.2852022647857666
Epoch  97  time  1.2600393295288086
Epoch  98  time  1.2886483669281006
Epoch  99  time  1.5262267589569092
Epoch  100  time  1.7609002590179443
Epoch  100  loss  2.055715019880463 correct 47
Epoch  101  time  1.2468230724334717
Epoch  102  time  1.2496907711029053
Epoch  103  time  1.3129935264587402
Epoch  104  time  1.2540841102600098
Epoch  105  time  1.2753305435180664
Epoch  106  time  1.2727625370025635
Epoch  107  time  1.2949984073638916
Epoch  108  time  1.559976577758789
Epoch  109  time  1.7602975368499756
Epoch  110  time  1.2454447746276855
Epoch  110  loss  1.678392263258444 correct 49
Epoch  111  time  1.2904314994812012
Epoch  112  time  1.3508591651916504
Epoch  113  time  1.9464671611785889
Epoch  114  time  1.3229787349700928
Epoch  115  time  1.2408080101013184
Epoch  116  time  1.2466669082641602
Epoch  117  time  1.8421297073364258
Epoch  118  time  1.4963982105255127
Epoch  119  time  1.2406055927276611
Epoch  120  time  1.245924711227417
Epoch  120  loss  1.374162901683828 correct 47
Epoch  121  time  1.2463955879211426
Epoch  122  time  1.2528951168060303
Epoch  123  time  1.2325680255889893
Epoch  124  time  1.2944104671478271
Epoch  125  time  1.256788969039917
Epoch  126  time  1.7748448848724365
Epoch  127  time  1.5209760665893555
Epoch  128  time  1.2654166221618652
Epoch  129  time  1.2960615158081055
Epoch  130  time  1.2477879524230957
Epoch  130  loss  2.428958735258055 correct 49
Epoch  131  time  1.2519011497497559
Epoch  132  time  1.2379460334777832
Epoch  133  time  1.2651467323303223
Epoch  134  time  1.296722412109375
Epoch  135  time  1.6980857849121094
Epoch  136  time  1.5813543796539307
Epoch  137  time  1.2554388046264648
Epoch  138  time  1.2904062271118164
Epoch  139  time  1.306504249572754
Epoch  140  time  1.2560982704162598
Epoch  140  loss  1.8110190618338855 correct 50
Epoch  141  time  1.2765398025512695
Epoch  142  time  1.2420639991760254
Epoch  143  time  1.2524077892303467
Epoch  144  time  1.6970136165618896
Epoch  145  time  1.6124069690704346
Epoch  146  time  1.2482109069824219
Epoch  147  time  1.259364366531372
Epoch  148  time  1.2624757289886475
Epoch  149  time  1.3066177368164062
Epoch  150  time  1.2519757747650146
Epoch  150  loss  1.7089120186798596 correct 49
Epoch  151  time  1.2843079566955566
Epoch  152  time  1.2387752532958984
Epoch  153  time  1.5914416313171387
Epoch  154  time  1.744933843612671
Epoch  155  time  1.2548301219940186
Epoch  156  time  1.2596216201782227
Epoch  157  time  1.2474656105041504
Epoch  158  time  1.2401707172393799
Epoch  159  time  1.2981970310211182
Epoch  160  time  1.239544153213501
Epoch  160  loss  0.9583124602241441 correct 50
Epoch  161  time  1.243168592453003
Epoch  162  time  1.517885446548462
Epoch  163  time  1.7800467014312744
Epoch  164  time  1.2895426750183105
Epoch  165  time  1.2536101341247559
Epoch  166  time  1.2556843757629395
Epoch  167  time  1.244835376739502
Epoch  168  time  1.246239185333252
Epoch  169  time  1.2995779514312744
Epoch  170  time  1.2739543914794922
Epoch  170  loss  1.2459260451479373 correct 49
Epoch  171  time  1.468719482421875
Epoch  172  time  1.833604335784912
Epoch  173  time  1.2904918193817139
Epoch  174  time  1.2958595752716064
Epoch  175  time  1.247361421585083
Epoch  176  time  1.2616627216339111
Epoch  177  time  1.2421789169311523
Epoch  178  time  1.253843069076538
Epoch  179  time  1.2941632270812988
Epoch  180  time  1.4047491550445557
Epoch  180  loss  1.7776235956896231 correct 48
Epoch  181  time  1.8666670322418213
Epoch  182  time  1.2695586681365967
Epoch  183  time  1.2720401287078857
Epoch  184  time  1.2949597835540771
Epoch  185  time  1.2654473781585693
Epoch  186  time  1.2535645961761475
Epoch  187  time  1.2472949028015137
Epoch  188  time  1.247732400894165
Epoch  189  time  1.4136312007904053
Epoch  190  time  1.8636138439178467
Epoch  190  loss  1.4936734874720339 correct 50
Epoch  191  time  1.2517719268798828
Epoch  192  time  1.253772258758545
Epoch  193  time  1.2448995113372803
Epoch  194  time  1.3132884502410889
Epoch  195  time  1.289292573928833
Epoch  196  time  1.2566015720367432
Epoch  197  time  1.2462759017944336
Epoch  198  time  1.2949490547180176
Epoch  199  time  1.9253311157226562
Epoch  200  time  1.3768327236175537
Epoch  200  loss  0.46416722410332356 correct 50
Epoch  201  time  1.238795518875122
Epoch  202  time  1.2371156215667725
Epoch  203  time  1.2573649883270264
Epoch  204  time  1.3197755813598633
Epoch  205  time  1.2422430515289307
Epoch  206  time  1.2403838634490967
Epoch  207  time  1.2460017204284668
Epoch  208  time  1.8476932048797607
Epoch  209  time  1.5065970420837402
Epoch  210  time  1.2638659477233887
Epoch  210  loss  0.47453079113207447 correct 49
Epoch  211  time  1.235673427581787
Epoch  212  time  1.2416129112243652
Epoch  213  time  1.2485864162445068
Epoch  214  time  1.3071503639221191
Epoch  215  time  1.2817621231079102
Epoch  216  time  1.2518999576568604
Epoch  217  time  1.8042802810668945
Epoch  218  time  1.6224725246429443
Epoch  219  time  1.3134753704071045
Epoch  220  time  1.2608973979949951
Epoch  220  loss  0.658692621406719 correct 50
Epoch  221  time  1.2436683177947998
Epoch  222  time  1.2781190872192383
Epoch  223  time  1.2558205127716064
Epoch  224  time  1.3006255626678467
Epoch  225  time  1.2459559440612793
Epoch  226  time  1.7070462703704834
Epoch  227  time  1.6312315464019775
Epoch  228  time  1.2630207538604736
Epoch  229  time  1.2982933521270752
Epoch  230  time  1.2555227279663086
Epoch  230  loss  0.5185362712095062 correct 49
Epoch  231  time  1.2557778358459473
Epoch  232  time  1.264207363128662
Epoch  233  time  1.2582049369812012
Epoch  234  time  1.293100357055664
Epoch  235  time  1.6302947998046875
Epoch  236  time  1.6566119194030762
Epoch  237  time  1.270552396774292
Epoch  238  time  1.2650043964385986
Epoch  239  time  1.2929558753967285
Epoch  240  time  1.2856965065002441
Epoch  240  loss  0.4658564787404925 correct 50
Epoch  241  time  1.2340517044067383
Epoch  242  time  1.25614333152771
Epoch  243  time  1.2554879188537598
Epoch  244  time  1.6193010807037354
Epoch  245  time  1.7222926616668701
Epoch  246  time  1.2490367889404297
Epoch  247  time  1.249115228652954
Epoch  248  time  1.2571609020233154
Epoch  249  time  1.2935638427734375
Epoch  250  time  1.243089199066162
Epoch  250  loss  1.4018649149639408 correct 50
Epoch  251  time  1.2399303913116455
Epoch  252  time  1.2476940155029297
Epoch  253  time  1.4593207836151123
Epoch  254  time  1.8613865375518799
Epoch  255  time  1.2493510246276855
Epoch  256  time  1.2485897541046143
Epoch  257  time  1.250166416168213
Epoch  258  time  1.2550487518310547
Epoch  259  time  1.324662208557129
Epoch  260  time  1.246910572052002
Epoch  260  loss  0.4119320147244074 correct 50
Epoch  261  time  1.233992338180542
Epoch  262  time  1.4065518379211426
Epoch  263  time  1.9009997844696045
Epoch  264  time  1.3326969146728516
Epoch  265  time  1.2434895038604736
Epoch  266  time  1.2456786632537842
Epoch  267  time  1.2513782978057861
Epoch  268  time  1.2393314838409424
Epoch  269  time  1.3033950328826904
Epoch  270  time  1.2447443008422852
Epoch  270  loss  0.5611594212623332 correct 50
Epoch  271  time  1.3841748237609863
Epoch  272  time  1.842390775680542
Epoch  273  time  1.3670194149017334
Epoch  274  time  1.2930364608764648
Epoch  275  time  1.2436697483062744
Epoch  276  time  1.2621924877166748
Epoch  277  time  1.2513511180877686
Epoch  278  time  1.2406487464904785
Epoch  279  time  1.3123934268951416
Epoch  280  time  1.245572805404663
Epoch  280  loss  0.7822283227451119 correct 49
Epoch  281  time  1.8663384914398193
Epoch  282  time  1.453465461730957
Epoch  283  time  1.2630183696746826
Epoch  284  time  1.3046188354492188
Epoch  285  time  1.256007194519043
Epoch  286  time  1.2667796611785889
Epoch  287  time  1.2464261054992676
Epoch  288  time  1.242365837097168
Epoch  289  time  1.270418405532837
Epoch  290  time  1.8150947093963623
Epoch  290  loss  0.18593429121108074 correct 50
Epoch  291  time  1.4816741943359375
Epoch  292  time  1.249675989151001
Epoch  293  time  1.2481622695922852
Epoch  294  time  1.2582378387451172
Epoch  295  time  1.288442611694336
Epoch  296  time  1.249730110168457
Epoch  297  time  1.2364821434020996
Epoch  298  time  1.2380504608154297
Epoch  299  time  1.6103408336639404
Epoch  300  time  1.6953330039978027
Epoch  300  loss  0.7349992591252246 correct 50
Epoch  301  time  1.2737300395965576
Epoch  302  time  1.2421934604644775
Epoch  303  time  1.2487168312072754
Epoch  304  time  1.237595558166504
Epoch  305  time  1.2974262237548828
Epoch  306  time  1.2669432163238525
Epoch  307  time  1.272557020187378
Epoch  308  time  1.5825188159942627
Epoch  309  time  1.783257246017456
Epoch  310  time  1.2462425231933594
Epoch  310  loss  1.2561196206062775 correct 50
Epoch  311  time  1.2513706684112549
Epoch  312  time  1.2429661750793457
Epoch  313  time  1.242978811264038
Epoch  314  time  1.2992515563964844
Epoch  315  time  1.2716593742370605
Epoch  316  time  1.2521185874938965
Epoch  317  time  1.5028600692749023
Epoch  318  time  1.7799301147460938
Epoch  319  time  1.2980475425720215
Epoch  320  time  1.2641849517822266
Epoch  320  loss  0.06083697440472306 correct 50
Epoch  321  time  1.2669081687927246
Epoch  322  time  1.238675594329834
Epoch  323  time  1.2453258037567139
Epoch  324  time  1.3141083717346191
Epoch  325  time  1.2573049068450928
Epoch  326  time  1.4638350009918213
Epoch  327  time  1.8248546123504639
Epoch  328  time  1.2410917282104492
Epoch  329  time  1.2670366764068604
Epoch  330  time  1.3059518337249756
Epoch  330  loss  0.8107698066894522 correct 50
Epoch  331  time  1.2536942958831787
Epoch  332  time  1.2549359798431396
Epoch  333  time  1.2450332641601562
Epoch  334  time  1.2462410926818848
Epoch  335  time  1.8499939441680908
Epoch  336  time  2.1139819622039795
Epoch  337  time  1.5084691047668457
Epoch  338  time  1.253382921218872
Epoch  339  time  1.238088846206665
Epoch  340  time  1.2902607917785645
Epoch  340  loss  0.09514954741540689 correct 50
Epoch  341  time  1.2554645538330078
Epoch  342  time  1.2477164268493652
Epoch  343  time  1.241062879562378
Epoch  344  time  1.247443437576294
Epoch  345  time  1.775428295135498
Epoch  346  time  1.5828032493591309
Epoch  347  time  1.2607736587524414
Epoch  348  time  1.2403147220611572
Epoch  349  time  1.2370519638061523
Epoch  350  time  1.2954919338226318
Epoch  350  loss  0.25825639018393004 correct 50
Epoch  351  time  1.2896616458892822
Epoch  352  time  1.2422528266906738
Epoch  353  time  1.2570295333862305
Epoch  354  time  1.645836591720581
Epoch  355  time  1.6931543350219727
Epoch  356  time  1.2348573207855225
Epoch  357  time  1.249096155166626
Epoch  358  time  1.242100715637207
Epoch  359  time  1.2789685726165771
Epoch  360  time  1.2981855869293213
Epoch  360  loss  0.20435613424342042 correct 50
Epoch  361  time  1.2768197059631348
Epoch  362  time  1.246039628982544
Epoch  363  time  1.567293643951416
Epoch  364  time  1.727670669555664
Epoch  365  time  1.287285566329956
Epoch  366  time  1.244971752166748
Epoch  367  time  1.243595838546753
Epoch  368  time  1.2500536441802979
Epoch  369  time  1.2440142631530762
Epoch  370  time  1.3057055473327637
Epoch  370  loss  0.21474979381644055 correct 50
Epoch  371  time  1.2512409687042236
Epoch  372  time  1.4738495349884033
Epoch  373  time  1.8259267807006836
Epoch  374  time  1.2339379787445068
Epoch  375  time  1.302483320236206
Epoch  376  time  1.251544713973999
Epoch  377  time  1.2444219589233398
Epoch  378  time  1.2514114379882812
Epoch  379  time  1.2431449890136719
Epoch  380  time  1.2933847904205322
Epoch  380  loss  0.6158300365763062 correct 50
Epoch  381  time  1.390690565109253
Epoch  382  time  1.8607637882232666
Epoch  383  time  1.2649688720703125
Epoch  384  time  1.2351183891296387
Epoch  385  time  1.2895872592926025
Epoch  386  time  1.2475521564483643
Epoch  387  time  1.2379381656646729
Epoch  388  time  1.2309238910675049
Epoch  389  time  1.2372910976409912
Epoch  390  time  1.2917430400848389
Epoch  390  loss  0.1478543642314137 correct 50
Epoch  391  time  1.876802921295166
Epoch  392  time  1.3837709426879883
Epoch  393  time  1.2325878143310547
Epoch  394  time  1.2292304039001465
Epoch  395  time  1.3238942623138428
Epoch  396  time  1.2746880054473877
Epoch  397  time  1.2420308589935303
Epoch  398  time  1.2542569637298584
Epoch  399  time  1.2482521533966064
Epoch  400  time  1.7467148303985596
Epoch  400  loss  0.04335002989816451 correct 50
Epoch  401  time  1.5498840808868408
Epoch  402  time  1.2505834102630615
Epoch  403  time  1.263643503189087
Epoch  404  time  1.2428221702575684
Epoch  405  time  1.2459626197814941
Epoch  406  time  1.3177928924560547
Epoch  407  time  1.2358977794647217
Epoch  408  time  1.2487547397613525
Epoch  409  time  1.643547534942627
Epoch  410  time  1.6138505935668945
Epoch  410  loss  0.5920312705587523 correct 50
Epoch  411  time  1.3266470432281494
Epoch  412  time  1.2369272708892822
Epoch  413  time  1.261204481124878
Epoch  414  time  1.2574543952941895
Epoch  415  time  1.2404425144195557
Epoch  416  time  1.3028874397277832
Epoch  417  time  1.2571642398834229
Epoch  418  time  1.5948431491851807
Epoch  419  time  1.7034218311309814
Epoch  420  time  1.253171443939209
Epoch  420  loss  0.3780386385563729 correct 50
Epoch  421  time  1.3120925426483154
Epoch  422  time  1.2404870986938477
Epoch  423  time  1.2352938652038574
Epoch  424  time  1.240605354309082
Epoch  425  time  1.2422430515289307
Epoch  426  time  1.305835247039795
Epoch  427  time  1.4997828006744385
Epoch  428  time  1.7920749187469482
Epoch  429  time  1.246436595916748
Epoch  430  time  1.2442080974578857
Epoch  430  loss  0.3549242818795967 correct 50
Epoch  431  time  1.3035428524017334
Epoch  432  time  1.2405521869659424
Epoch  433  time  1.2419772148132324
Epoch  434  time  1.2414357662200928
Epoch  435  time  1.2587080001831055
Epoch  436  time  1.5033841133117676
Epoch  437  time  1.8388361930847168
Epoch  438  time  1.2413735389709473
Epoch  439  time  1.2579448223114014
Epoch  440  time  1.2619867324829102
Epoch  440  loss  0.8770853639696713 correct 50
Epoch  441  time  1.3063898086547852
Epoch  442  time  1.2447330951690674
Epoch  443  time  1.244551420211792
Epoch  444  time  1.2438154220581055
Epoch  445  time  1.342008352279663
Epoch  446  time  1.9676342010498047
Epoch  447  time  1.2985000610351562
Epoch  448  time  1.2609355449676514
Epoch  449  time  1.2368290424346924
Epoch  450  time  1.259805679321289
Epoch  450  loss  0.16431812903228965 correct 50
Epoch  451  time  1.3095946311950684
Epoch  452  time  1.2483665943145752
Epoch  453  time  1.2492377758026123
Epoch  454  time  1.285945177078247
Epoch  455  time  1.8359661102294922
Epoch  456  time  1.4653511047363281
Epoch  457  time  1.2456514835357666
Epoch  458  time  1.26165771484375
Epoch  459  time  1.2556817531585693
Epoch  460  time  1.250758409500122
Epoch  460  loss  0.799739506907871 correct 50
Epoch  461  time  1.2897498607635498
Epoch  462  time  1.2425670623779297
Epoch  463  time  1.2640442848205566
Epoch  464  time  1.8104248046875
Epoch  465  time  1.4654700756072998
Epoch  466  time  1.3105957508087158
Epoch  467  time  1.23490309715271
Epoch  468  time  1.2443368434906006
Epoch  469  time  1.2399516105651855
Epoch  470  time  1.2464704513549805
Epoch  470  loss  0.14282750780388107 correct 50
Epoch  471  time  1.2873551845550537
Epoch  472  time  1.257427453994751
Epoch  473  time  1.7127227783203125
Epoch  474  time  1.5773875713348389
Epoch  475  time  1.2325012683868408
Epoch  476  time  1.3090777397155762
Epoch  477  time  1.2461769580841064
Epoch  478  time  1.2320420742034912
Epoch  479  time  1.2324535846710205
Epoch  480  time  1.2590007781982422
Epoch  480  loss  0.4932570511380481 correct 50
Epoch  481  time  1.2938494682312012
Epoch  482  time  1.5878198146820068
Epoch  483  time  1.6725399494171143
Epoch  484  time  1.2320747375488281
Epoch  485  time  1.2636077404022217
Epoch  486  time  1.2509007453918457
Epoch  487  time  1.2924485206604004
Epoch  488  time  1.2560186386108398
Epoch  489  time  1.241283655166626
Epoch  490  time  1.2450096607208252
Epoch  490  loss  0.11279921513939595 correct 50
Epoch  491  time  1.4860103130340576
Epoch  492  time  1.8505423069000244
Epoch  493  time  1.2430312633514404
Epoch  494  time  1.2457692623138428
Epoch  495  time  1.245678186416626
Epoch  496  time  1.259162425994873
Epoch  497  time  1.300053596496582
Epoch  498  time  1.2520594596862793
Epoch  499  time  1.2403647899627686
```
</details>


python run_fast_tensor.py --BACKEND cpu --HIDDEN 100 --DATASET xor --RATE 0.05

*Average time per epoch 0.18055780792236328 (for 500 epochs)*

<details>
<summary>View log</summary>
Epoch  0  time  5.784484148025513
Epoch  0  loss  7.202824217501043 correct 26
Epoch  1  time  0.17755484580993652
Epoch  2  time  0.16792798042297363
Epoch  3  time  0.18068885803222656
Epoch  4  time  0.1590721607208252
Epoch  5  time  0.16526317596435547
Epoch  6  time  0.16129207611083984
Epoch  7  time  0.20291900634765625
Epoch  8  time  0.1620948314666748
Epoch  9  time  0.16358709335327148
Epoch  10  time  0.16103506088256836
Epoch  10  loss  5.029012100679447 correct 46
Epoch  11  time  0.16762089729309082
Epoch  12  time  0.16528105735778809
Epoch  13  time  0.16393208503723145
Epoch  14  time  0.16640090942382812
Epoch  15  time  0.1646580696105957
Epoch  16  time  0.16362595558166504
Epoch  17  time  0.1642780303955078
Epoch  18  time  0.1673429012298584
Epoch  19  time  0.16139006614685059
Epoch  20  time  0.18035411834716797
Epoch  20  loss  3.622467963962505 correct 46
Epoch  21  time  0.16495084762573242
Epoch  22  time  0.1645655632019043
Epoch  23  time  0.16411972045898438
Epoch  24  time  0.16354990005493164
Epoch  25  time  0.17022991180419922
Epoch  26  time  0.17344093322753906
Epoch  27  time  0.16450905799865723
Epoch  28  time  0.16578221321105957
Epoch  29  time  0.16331219673156738
Epoch  30  time  0.16504406929016113
Epoch  30  loss  4.876344166632235 correct 45
Epoch  31  time  0.16406607627868652
Epoch  32  time  0.16512799263000488
Epoch  33  time  0.15853309631347656
Epoch  34  time  0.17422103881835938
Epoch  35  time  0.16626477241516113
Epoch  36  time  0.15989303588867188
Epoch  37  time  0.1596231460571289
Epoch  38  time  0.1634819507598877
Epoch  39  time  0.1608867645263672
Epoch  40  time  0.1633448600769043
Epoch  40  loss  4.45227922646889 correct 44
Epoch  41  time  0.1607217788696289
Epoch  42  time  0.16382098197937012
Epoch  43  time  0.1657559871673584
Epoch  44  time  0.16494393348693848
Epoch  45  time  0.1623697280883789
Epoch  46  time  0.1653590202331543
Epoch  47  time  0.1660292148590088
Epoch  48  time  0.17145299911499023
Epoch  49  time  0.1654829978942871
Epoch  50  time  0.165510892868042
Epoch  50  loss  1.985876524747854 correct 47
Epoch  51  time  0.16377902030944824
Epoch  52  time  0.17016816139221191
Epoch  53  time  0.16623401641845703
Epoch  54  time  0.16498398780822754
Epoch  55  time  0.17638611793518066
Epoch  56  time  0.16190791130065918
Epoch  57  time  0.16048097610473633
Epoch  58  time  0.16431188583374023
Epoch  59  time  0.1631932258605957
Epoch  60  time  0.16407394409179688
Epoch  60  loss  1.4167897228646624 correct 46
Epoch  61  time  0.16161680221557617
Epoch  62  time  0.1651611328125
Epoch  63  time  0.17824101448059082
Epoch  64  time  0.1708829402923584
Epoch  65  time  0.17314386367797852
Epoch  66  time  0.16263508796691895
Epoch  67  time  0.15935373306274414
Epoch  68  time  0.17009711265563965
Epoch  69  time  0.16591596603393555
Epoch  70  time  0.1643691062927246
Epoch  70  loss  3.938496397394098 correct 45
Epoch  71  time  0.16806888580322266
Epoch  72  time  0.1575782299041748
Epoch  73  time  0.1650838851928711
Epoch  74  time  0.18175888061523438
Epoch  75  time  0.16630101203918457
Epoch  76  time  0.16418027877807617
Epoch  77  time  0.16043400764465332
Epoch  78  time  0.16212892532348633
Epoch  79  time  0.16953611373901367
Epoch  80  time  0.1712932586669922
Epoch  80  loss  1.7319049018135442 correct 46
Epoch  81  time  0.17345094680786133
Epoch  82  time  0.1709001064300537
Epoch  83  time  0.17142605781555176
Epoch  84  time  0.1663520336151123
Epoch  85  time  0.16945505142211914
Epoch  86  time  0.16269803047180176
Epoch  87  time  0.16185212135314941
Epoch  88  time  0.16231822967529297
Epoch  89  time  0.1618950366973877
Epoch  90  time  0.1622917652130127
Epoch  90  loss  0.8073675852439954 correct 47
Epoch  91  time  0.16631388664245605
Epoch  92  time  0.16729998588562012
Epoch  93  time  0.17049598693847656
Epoch  94  time  0.16080784797668457
Epoch  95  time  0.17495179176330566
Epoch  96  time  0.1600508689880371
Epoch  97  time  0.16415095329284668
Epoch  98  time  0.16097712516784668
Epoch  99  time  0.16316986083984375
Epoch  100  time  0.16350817680358887
Epoch  100  loss  3.3149157256177215 correct 46
Epoch  101  time  0.15793299674987793
Epoch  102  time  0.16509485244750977
Epoch  103  time  0.17037200927734375
Epoch  104  time  0.16302227973937988
Epoch  105  time  0.16427111625671387
Epoch  106  time  0.1648390293121338
Epoch  107  time  0.16374707221984863
Epoch  108  time  0.16779804229736328
Epoch  109  time  0.16602873802185059
Epoch  110  time  0.1595621109008789
Epoch  110  loss  1.021779464275225 correct 47
Epoch  111  time  0.15665793418884277
Epoch  112  time  0.1663191318511963
Epoch  113  time  0.16686582565307617
Epoch  114  time  0.16289210319519043
Epoch  115  time  0.20084404945373535
Epoch  116  time  0.16291260719299316
Epoch  117  time  0.1956329345703125
Epoch  118  time  0.16910886764526367
Epoch  119  time  0.17178130149841309
Epoch  120  time  0.1581869125366211
Epoch  120  loss  1.7311481863790832 correct 48
Epoch  121  time  0.1680588722229004
Epoch  122  time  0.1877760887145996
Epoch  123  time  0.1725161075592041
Epoch  124  time  0.17618799209594727
Epoch  125  time  0.17792606353759766
Epoch  126  time  0.17067503929138184
Epoch  127  time  0.20518112182617188
Epoch  128  time  0.19545388221740723
Epoch  129  time  0.19999003410339355
Epoch  130  time  0.17946767807006836
Epoch  130  loss  1.3253064350233 correct 48
Epoch  131  time  0.18383097648620605
Epoch  132  time  0.1707901954650879
Epoch  133  time  0.19733572006225586
Epoch  134  time  0.16574311256408691
Epoch  135  time  0.16950201988220215
Epoch  136  time  0.16495990753173828
Epoch  137  time  0.1658189296722412
Epoch  138  time  0.17148184776306152
Epoch  139  time  0.16698598861694336
Epoch  140  time  0.16327691078186035
Epoch  140  loss  1.8372316625640526 correct 49
Epoch  141  time  0.16592192649841309
Epoch  142  time  0.1675121784210205
Epoch  143  time  0.17535114288330078
Epoch  144  time  0.16948604583740234
Epoch  145  time  0.16433095932006836
Epoch  146  time  0.16652393341064453
Epoch  147  time  0.15357017517089844
Epoch  148  time  0.15589594841003418
Epoch  149  time  0.160552978515625
Epoch  150  time  0.16409087181091309
Epoch  150  loss  2.0046965237299688 correct 49
Epoch  151  time  0.16016197204589844
Epoch  152  time  0.1644752025604248
Epoch  153  time  0.16015219688415527
Epoch  154  time  0.1721658706665039
Epoch  155  time  0.17116379737854004
Epoch  156  time  0.15824294090270996
Epoch  157  time  0.16884899139404297
Epoch  158  time  0.16429901123046875
Epoch  159  time  0.16171908378601074
Epoch  160  time  0.1630101203918457
Epoch  160  loss  1.1855808917340511 correct 49
Epoch  161  time  0.16074705123901367
Epoch  162  time  0.16555190086364746
Epoch  163  time  0.16617608070373535
Epoch  164  time  0.16634416580200195
Epoch  165  time  0.24346399307250977
Epoch  166  time  0.2328798770904541
Epoch  167  time  0.17526912689208984
Epoch  168  time  0.15799999237060547
Epoch  169  time  0.16201400756835938
Epoch  170  time  0.16291308403015137
Epoch  170  loss  1.0035958057205996 correct 49
Epoch  171  time  0.16117405891418457
Epoch  172  time  0.17184710502624512
Epoch  173  time  0.16833114624023438
Epoch  174  time  0.15969085693359375
Epoch  175  time  0.16472077369689941
Epoch  176  time  0.19139623641967773
Epoch  177  time  0.16637492179870605
Epoch  178  time  0.16663217544555664
Epoch  179  time  0.18433308601379395
Epoch  180  time  0.16397738456726074
Epoch  180  loss  0.6446407273899369 correct 49
Epoch  181  time  0.16522598266601562
Epoch  182  time  0.1634988784790039
Epoch  183  time  0.16524386405944824
Epoch  184  time  0.17223000526428223
Epoch  185  time  0.1665811538696289
Epoch  186  time  0.16735506057739258
Epoch  187  time  0.16623306274414062
Epoch  188  time  0.16082501411437988
Epoch  189  time  0.18128299713134766
Epoch  190  time  0.17657804489135742
Epoch  190  loss  1.5409656650156487 correct 49
Epoch  191  time  0.1675703525543213
Epoch  192  time  0.1602158546447754
Epoch  193  time  0.163222074508667
Epoch  194  time  0.16412997245788574
Epoch  195  time  0.1650390625
Epoch  196  time  0.16242313385009766
Epoch  197  time  0.16060209274291992
Epoch  198  time  0.16234374046325684
Epoch  199  time  0.1668078899383545
Epoch  200  time  0.16373395919799805
Epoch  200  loss  0.8147587158519036 correct 49
Epoch  201  time  0.17635369300842285
Epoch  202  time  0.16153717041015625
Epoch  203  time  0.16523289680480957
Epoch  204  time  0.16424798965454102
Epoch  205  time  0.17149972915649414
Epoch  206  time  0.16648006439208984
Epoch  207  time  0.16704106330871582
Epoch  208  time  0.1722118854522705
Epoch  209  time  0.16926121711730957
Epoch  210  time  0.16582489013671875
Epoch  210  loss  1.0210312691166028 correct 49
Epoch  211  time  0.1669178009033203
Epoch  212  time  0.17169690132141113
Epoch  213  time  0.16545510292053223
Epoch  214  time  0.1672070026397705
Epoch  215  time  0.1731100082397461
Epoch  216  time  0.16038990020751953
Epoch  217  time  0.1758880615234375
Epoch  218  time  0.15955185890197754
Epoch  219  time  0.1617598533630371
Epoch  220  time  0.16881895065307617
Epoch  220  loss  0.375117923042378 correct 50
Epoch  221  time  0.16486310958862305
Epoch  222  time  0.1659698486328125
Epoch  223  time  0.17768287658691406
Epoch  224  time  0.16290616989135742
Epoch  225  time  0.16668486595153809
Epoch  226  time  0.17073893547058105
Epoch  227  time  0.17151474952697754
Epoch  228  time  0.16727614402770996
Epoch  229  time  0.16201210021972656
Epoch  230  time  0.16159296035766602
Epoch  230  loss  1.1746172724473714 correct 49
Epoch  231  time  0.16684317588806152
Epoch  232  time  0.1600949764251709
Epoch  233  time  0.15903902053833008
Epoch  234  time  0.16141200065612793
Epoch  235  time  0.16053080558776855
Epoch  236  time  0.15517401695251465
Epoch  237  time  0.1572129726409912
Epoch  238  time  0.15992498397827148
Epoch  239  time  0.16148018836975098
Epoch  240  time  0.15886425971984863
Epoch  240  loss  0.6420884804414806 correct 49
Epoch  241  time  0.16623210906982422
Epoch  242  time  0.16770100593566895
Epoch  243  time  0.1660451889038086
Epoch  244  time  0.17037272453308105
Epoch  245  time  0.16173696517944336
Epoch  246  time  0.15792012214660645
Epoch  247  time  0.1637740135192871
Epoch  248  time  0.16403889656066895
Epoch  249  time  0.16410017013549805
Epoch  250  time  0.17032384872436523
Epoch  250  loss  0.29789467088868704 correct 50
Epoch  251  time  0.17069697380065918
Epoch  252  time  0.16609406471252441
Epoch  253  time  0.16530609130859375
Epoch  254  time  0.17479181289672852
Epoch  255  time  0.15748906135559082
Epoch  256  time  0.1600959300994873
Epoch  257  time  0.16389012336730957
Epoch  258  time  0.17070794105529785
Epoch  259  time  0.16297602653503418
Epoch  260  time  0.1593921184539795
Epoch  260  loss  0.3643433069754876 correct 50
Epoch  261  time  0.16950201988220215
Epoch  262  time  0.16675925254821777
Epoch  263  time  0.17078804969787598
Epoch  264  time  0.16127371788024902
Epoch  265  time  0.16485118865966797
Epoch  266  time  0.16098785400390625
Epoch  267  time  0.1595001220703125
Epoch  268  time  0.16472196578979492
Epoch  269  time  0.16683387756347656
Epoch  270  time  0.16576790809631348
Epoch  270  loss  1.8951649556266583 correct 49
Epoch  271  time  0.16330504417419434
Epoch  272  time  0.1668858528137207
Epoch  273  time  0.17187190055847168
Epoch  274  time  0.16548895835876465
Epoch  275  time  0.16945576667785645
Epoch  276  time  0.1636369228363037
Epoch  277  time  0.15645384788513184
Epoch  278  time  0.16274714469909668
Epoch  279  time  0.16460800170898438
Epoch  280  time  0.16066694259643555
Epoch  280  loss  0.33202362296646415 correct 49
Epoch  281  time  0.1658940315246582
Epoch  282  time  0.1661229133605957
Epoch  283  time  0.16184306144714355
Epoch  284  time  0.16244721412658691
Epoch  285  time  0.16623210906982422
Epoch  286  time  0.1656510829925537
Epoch  287  time  0.16515707969665527
Epoch  288  time  0.15836310386657715
Epoch  289  time  0.16551780700683594
Epoch  290  time  0.16620802879333496
Epoch  290  loss  1.1704536678603294 correct 50
Epoch  291  time  0.18579816818237305
Epoch  292  time  0.1568911075592041
Epoch  293  time  0.15914201736450195
Epoch  294  time  0.18246698379516602
Epoch  295  time  0.1620321273803711
Epoch  296  time  0.1703948974609375
Epoch  297  time  0.1623859405517578
Epoch  298  time  0.16574478149414062
Epoch  299  time  0.1708669662475586
Epoch  300  time  0.16254472732543945
Epoch  300  loss  0.15372720598170186 correct 49
Epoch  301  time  0.1568000316619873
Epoch  302  time  0.16306710243225098
Epoch  303  time  0.1620500087738037
Epoch  304  time  0.17116689682006836
Epoch  305  time  0.16351604461669922
Epoch  306  time  0.15801000595092773
Epoch  307  time  0.1683199405670166
Epoch  308  time  0.1601240634918213
Epoch  309  time  0.16558599472045898
Epoch  310  time  0.16176414489746094
Epoch  310  loss  1.5846083015024506 correct 49
Epoch  311  time  0.1759779453277588
Epoch  312  time  0.18278908729553223
Epoch  313  time  0.20673108100891113
Epoch  314  time  0.1708049774169922
Epoch  315  time  0.1733567714691162
Epoch  316  time  0.17215204238891602
Epoch  317  time  0.1699972152709961
Epoch  318  time  0.16362690925598145
Epoch  319  time  0.20123076438903809
Epoch  320  time  0.1857290267944336
Epoch  320  loss  0.7342644887397386 correct 49
Epoch  321  time  0.17125821113586426
Epoch  322  time  0.18520617485046387
Epoch  323  time  0.1748518943786621
Epoch  324  time  0.17736124992370605
Epoch  325  time  0.18192195892333984
Epoch  326  time  0.18279814720153809
Epoch  327  time  0.17215991020202637
Epoch  328  time  0.16468286514282227
Epoch  329  time  0.18097305297851562
Epoch  330  time  0.2001791000366211
Epoch  330  loss  0.06228376087636335 correct 49
Epoch  331  time  0.1940779685974121
Epoch  332  time  0.1862330436706543
Epoch  333  time  0.1624300479888916
Epoch  334  time  0.17328286170959473
Epoch  335  time  0.17341399192810059
Epoch  336  time  0.17627906799316406
Epoch  337  time  0.17749714851379395
Epoch  338  time  0.17084503173828125
Epoch  339  time  0.1640939712524414
Epoch  340  time  0.1736457347869873
Epoch  340  loss  1.2909941428435223 correct 50
Epoch  341  time  0.20901203155517578
Epoch  342  time  0.17248010635375977
Epoch  343  time  0.17750191688537598
Epoch  344  time  0.17538189888000488
Epoch  345  time  0.18047714233398438
Epoch  346  time  0.16921710968017578
Epoch  347  time  0.17707395553588867
Epoch  348  time  0.16487383842468262
Epoch  349  time  0.23726987838745117
Epoch  350  time  0.16954803466796875
Epoch  350  loss  1.4944484208339317 correct 49
Epoch  351  time  0.19582796096801758
Epoch  352  time  0.17502307891845703
Epoch  353  time  0.16860389709472656
Epoch  354  time  0.18387985229492188
Epoch  355  time  0.16943693161010742
Epoch  356  time  0.1827080249786377
Epoch  357  time  0.1730968952178955
Epoch  358  time  0.17763423919677734
Epoch  359  time  0.1650848388671875
Epoch  360  time  0.16691994667053223
Epoch  360  loss  0.045961839842293614 correct 49
Epoch  361  time  0.16277766227722168
Epoch  362  time  0.17722511291503906
Epoch  363  time  0.1960451602935791
Epoch  364  time  0.19338202476501465
Epoch  365  time  0.17748618125915527
Epoch  366  time  0.1703019142150879
Epoch  367  time  0.18332314491271973
Epoch  368  time  0.18423700332641602
Epoch  369  time  0.1621870994567871
Epoch  370  time  0.17046403884887695
Epoch  370  loss  0.4028406232848007 correct 50
Epoch  371  time  0.16023492813110352
Epoch  372  time  0.1743788719177246
Epoch  373  time  0.16641473770141602
Epoch  374  time  0.17426395416259766
Epoch  375  time  0.1781759262084961
Epoch  376  time  0.16997694969177246
Epoch  377  time  0.1695082187652588
Epoch  378  time  0.16708016395568848
Epoch  379  time  0.17725896835327148
Epoch  380  time  0.19124507904052734
Epoch  380  loss  0.1530252143415021 correct 49
Epoch  381  time  0.1888730525970459
Epoch  382  time  0.17970514297485352
Epoch  383  time  0.18266773223876953
Epoch  384  time  0.1947019100189209
Epoch  385  time  0.19615674018859863
Epoch  386  time  0.1799333095550537
Epoch  387  time  0.1694951057434082
Epoch  388  time  0.1785449981689453
Epoch  389  time  0.17684507369995117
Epoch  390  time  0.1742420196533203
Epoch  390  loss  0.18403254004073968 correct 50
Epoch  391  time  0.16647624969482422
Epoch  392  time  0.15939617156982422
Epoch  393  time  0.16582584381103516
Epoch  394  time  0.1645979881286621
Epoch  395  time  0.17326903343200684
Epoch  396  time  0.1766049861907959
Epoch  397  time  0.17014598846435547
Epoch  398  time  0.16737008094787598
Epoch  399  time  0.18016982078552246
Epoch  400  time  0.16756987571716309
Epoch  400  loss  0.3219043146538809 correct 49
Epoch  401  time  0.1827986240386963
Epoch  402  time  0.17667913436889648
Epoch  403  time  0.20902109146118164
Epoch  404  time  0.19174885749816895
Epoch  405  time  0.22452998161315918
Epoch  406  time  0.20036602020263672
Epoch  407  time  0.18966293334960938
Epoch  408  time  0.1851329803466797
Epoch  409  time  0.18503022193908691
Epoch  410  time  0.1753232479095459
Epoch  410  loss  1.10893764064647 correct 49
Epoch  411  time  0.16431212425231934
Epoch  412  time  0.1652059555053711
Epoch  413  time  0.15948796272277832
Epoch  414  time  0.16894197463989258
Epoch  415  time  0.17116713523864746
Epoch  416  time  0.16020679473876953
Epoch  417  time  0.17491412162780762
Epoch  418  time  0.1679990291595459
Epoch  419  time  0.166823148727417
Epoch  420  time  0.16305303573608398
Epoch  420  loss  0.4368049466006553 correct 50
Epoch  421  time  0.16033101081848145
Epoch  422  time  0.16517186164855957
Epoch  423  time  0.16667389869689941
Epoch  424  time  0.16654396057128906
Epoch  425  time  0.16563010215759277
Epoch  426  time  0.1588118076324463
Epoch  427  time  0.16695499420166016
Epoch  428  time  0.1654529571533203
Epoch  429  time  0.1632688045501709
Epoch  430  time  0.1632540225982666
Epoch  430  loss  1.0339311325989422 correct 49
Epoch  431  time  0.1798551082611084
Epoch  432  time  0.16116094589233398
Epoch  433  time  0.17186784744262695
Epoch  434  time  0.16069889068603516
Epoch  435  time  0.1668229103088379
Epoch  436  time  0.16597819328308105
Epoch  437  time  0.16038894653320312
Epoch  438  time  0.16507577896118164
Epoch  439  time  0.16209816932678223
Epoch  440  time  0.16349291801452637
Epoch  440  loss  0.3486020080855348 correct 49
Epoch  441  time  0.16431093215942383
Epoch  442  time  0.1592259407043457
Epoch  443  time  0.16922879219055176
Epoch  444  time  0.1604928970336914
Epoch  445  time  0.16295504570007324
Epoch  446  time  0.16614294052124023
Epoch  447  time  0.16986393928527832
Epoch  448  time  0.1640479564666748
Epoch  449  time  0.17260503768920898
Epoch  450  time  0.16552233695983887
Epoch  450  loss  0.7343388196859454 correct 49
Epoch  451  time  0.1594829559326172
Epoch  452  time  0.1673121452331543
Epoch  453  time  0.17059087753295898
Epoch  454  time  0.17013812065124512
Epoch  455  time  0.16756105422973633
Epoch  456  time  0.16611695289611816
Epoch  457  time  0.16924095153808594
Epoch  458  time  0.16343975067138672
Epoch  459  time  0.172288179397583
Epoch  460  time  0.16425871849060059
Epoch  460  loss  0.2520732108350773 correct 50
Epoch  461  time  0.17954802513122559
Epoch  462  time  0.16134119033813477
Epoch  463  time  0.1694939136505127
Epoch  464  time  0.16565299034118652
Epoch  465  time  0.16158604621887207
Epoch  466  time  0.1660919189453125
Epoch  467  time  0.16623306274414062
Epoch  468  time  0.16585302352905273
Epoch  469  time  0.16475820541381836
Epoch  470  time  0.16456007957458496
Epoch  470  loss  0.09359199242600288 correct 50
Epoch  471  time  0.1678469181060791
Epoch  472  time  0.16112303733825684
Epoch  473  time  0.16049885749816895
Epoch  474  time  0.1714949607849121
Epoch  475  time  0.16415691375732422
Epoch  476  time  0.1593470573425293
Epoch  477  time  0.16182708740234375
Epoch  478  time  0.15624618530273438
Epoch  479  time  0.17275023460388184
Epoch  480  time  0.1636669635772705
Epoch  480  loss  1.270962233507297 correct 49
Epoch  481  time  0.1864781379699707
Epoch  482  time  0.19208598136901855
Epoch  483  time  0.21439099311828613
Epoch  484  time  0.16875600814819336
Epoch  485  time  0.16708898544311523
Epoch  486  time  0.16162991523742676
Epoch  487  time  0.16126489639282227
Epoch  488  time  0.166856050491333
Epoch  489  time  0.1622929573059082
Epoch  490  time  0.17597007751464844
Epoch  490  loss  0.05826072733918435 correct 49
Epoch  491  time  0.16395187377929688
Epoch  492  time  0.16590094566345215
Epoch  493  time  0.16430902481079102
Epoch  494  time  0.16807794570922852
Epoch  495  time  0.1607367992401123
Epoch  496  time  0.16225790977478027
Epoch  497  time  0.15937399864196777
Epoch  498  time  0.1593761444091797
Epoch  499  time  0.16571807861328125
</details>

# Large Model

python run_fast_tensor.py --BACKEND cpu --HIDDEN 200 --DATASET xor --RATE 0.05

*Average time per epoch 0.5521484441757202 (for 500 epochs)*

<details>
<summary>View log</summary>
Epoch  0  time  6.960591077804565
Epoch  0  loss  20.103224284972445 correct 34
Epoch  1  time  0.5878419876098633
Epoch  2  time  0.5629191398620605
Epoch  3  time  0.5774831771850586
Epoch  4  time  0.5639839172363281
Epoch  5  time  0.5933640003204346
Epoch  6  time  0.599755048751831
Epoch  7  time  0.6122140884399414
Epoch  8  time  0.5885009765625
Epoch  9  time  0.5374128818511963
Epoch  10  time  0.5294680595397949
Epoch  10  loss  1.175162311065042 correct 46
Epoch  11  time  0.542884111404419
Epoch  12  time  0.5447251796722412
Epoch  13  time  0.5472061634063721
Epoch  14  time  0.5996830463409424
Epoch  15  time  0.5384080410003662
Epoch  16  time  0.5446288585662842
Epoch  17  time  0.5397448539733887
Epoch  18  time  0.5365631580352783
Epoch  19  time  0.5299227237701416
Epoch  20  time  0.5218329429626465
Epoch  20  loss  2.3817246570409267 correct 46
Epoch  21  time  0.5488851070404053
Epoch  22  time  0.5148210525512695
Epoch  23  time  0.5189900398254395
Epoch  24  time  0.5335960388183594
Epoch  25  time  0.519420862197876
Epoch  26  time  0.5438399314880371
Epoch  27  time  0.5358219146728516
Epoch  28  time  0.5266942977905273
Epoch  29  time  0.5195460319519043
Epoch  30  time  0.5832200050354004
Epoch  30  loss  0.2595669563172509 correct 47
Epoch  31  time  0.5468900203704834
Epoch  32  time  0.6921088695526123
Epoch  33  time  0.9590680599212646
Epoch  34  time  0.7258579730987549
Epoch  35  time  0.806818962097168
Epoch  36  time  0.9612071514129639
Epoch  37  time  0.5371520519256592
Epoch  38  time  0.5438969135284424
Epoch  39  time  0.5237686634063721
Epoch  40  time  0.5341048240661621
Epoch  40  loss  1.1060675685810315 correct 50
Epoch  41  time  0.5357048511505127
Epoch  42  time  0.5537500381469727
Epoch  43  time  0.6530852317810059
Epoch  44  time  0.5343410968780518
Epoch  45  time  0.5475988388061523
Epoch  46  time  0.5619959831237793
Epoch  47  time  0.5364489555358887
Epoch  48  time  0.5301680564880371
Epoch  49  time  0.5206968784332275
Epoch  50  time  0.5369858741760254
Epoch  50  loss  1.7164390961746068 correct 46
Epoch  51  time  0.5629661083221436
Epoch  52  time  0.5340609550476074
Epoch  53  time  0.5363759994506836
Epoch  54  time  0.5247838497161865
Epoch  55  time  0.5555520057678223
Epoch  56  time  0.5412790775299072
Epoch  57  time  0.5436089038848877
Epoch  58  time  0.5666089057922363
Epoch  59  time  0.5499939918518066
Epoch  60  time  0.5409131050109863
Epoch  60  loss  1.5996349259259919 correct 50
Epoch  61  time  0.5378139019012451
Epoch  62  time  0.5316569805145264
Epoch  63  time  0.5477230548858643
Epoch  64  time  0.5301482677459717
Epoch  65  time  0.5722858905792236
Epoch  66  time  0.5491039752960205
Epoch  67  time  0.5432190895080566
Epoch  68  time  0.5339717864990234
Epoch  69  time  0.7275629043579102
Epoch  70  time  0.521791934967041
Epoch  70  loss  0.5564110707209237 correct 48
Epoch  71  time  0.5338578224182129
Epoch  72  time  0.5252809524536133
Epoch  73  time  0.5196588039398193
Epoch  74  time  0.5211420059204102
Epoch  75  time  0.5317959785461426
Epoch  76  time  0.517164945602417
Epoch  77  time  0.5256702899932861
Epoch  78  time  0.5423250198364258
Epoch  79  time  0.5399680137634277
Epoch  80  time  0.5576059818267822
Epoch  80  loss  1.2742400541955272 correct 50
Epoch  81  time  0.5251529216766357
Epoch  82  time  0.516782283782959
Epoch  83  time  0.526202917098999
Epoch  84  time  0.5235700607299805
Epoch  85  time  0.5228500366210938
Epoch  86  time  0.5111513137817383
Epoch  87  time  0.5210926532745361
Epoch  88  time  0.5151371955871582
Epoch  89  time  0.5312371253967285
Epoch  90  time  0.5303428173065186
Epoch  90  loss  0.8121461421087592 correct 48
Epoch  91  time  0.5067741870880127
Epoch  92  time  0.5355720520019531
Epoch  93  time  0.5398039817810059
Epoch  94  time  0.5293970108032227
Epoch  95  time  0.5195631980895996
Epoch  96  time  0.5430240631103516
Epoch  97  time  0.5343837738037109
Epoch  98  time  0.5265069007873535
Epoch  99  time  0.5286438465118408
Epoch  100  time  0.5195860862731934
Epoch  100  loss  0.0614283686780726 correct 48
Epoch  101  time  0.5269901752471924
Epoch  102  time  0.5404250621795654
Epoch  103  time  0.5353100299835205
Epoch  104  time  0.5186560153961182
Epoch  105  time  0.5200791358947754
Epoch  106  time  0.5168776512145996
Epoch  107  time  0.5289649963378906
Epoch  108  time  0.5284421443939209
Epoch  109  time  0.5238137245178223
Epoch  110  time  0.5190868377685547
Epoch  110  loss  0.3205182631064708 correct 48
Epoch  111  time  0.543428897857666
Epoch  112  time  0.5240449905395508
Epoch  113  time  0.5166959762573242
Epoch  114  time  0.5054993629455566
Epoch  115  time  0.5182461738586426
Epoch  116  time  0.5225789546966553
Epoch  117  time  0.5213749408721924
Epoch  118  time  0.5168917179107666
Epoch  119  time  0.5119409561157227
Epoch  120  time  0.5275721549987793
Epoch  120  loss  0.7737701330810157 correct 48
Epoch  121  time  0.5329809188842773
Epoch  122  time  0.5106260776519775
Epoch  123  time  0.530839204788208
Epoch  124  time  0.5528240203857422
Epoch  125  time  0.5243480205535889
Epoch  126  time  0.5589771270751953
Epoch  127  time  0.5392751693725586
Epoch  128  time  0.515064001083374
Epoch  129  time  0.5310351848602295
Epoch  130  time  0.5268819332122803
Epoch  130  loss  0.04373904106520907 correct 48
Epoch  131  time  0.5594508647918701
Epoch  132  time  0.5448551177978516
Epoch  133  time  0.5426371097564697
Epoch  134  time  0.6040990352630615
Epoch  135  time  0.5650820732116699
Epoch  136  time  0.5592498779296875
Epoch  137  time  0.5606498718261719
Epoch  138  time  0.5732951164245605
Epoch  139  time  0.5484480857849121
Epoch  140  time  0.546332836151123
Epoch  140  loss  2.2400079041567325 correct 46
Epoch  141  time  0.5428979396820068
Epoch  142  time  0.5658280849456787
Epoch  143  time  0.5471539497375488
Epoch  144  time  0.5307769775390625
Epoch  145  time  0.5328922271728516
Epoch  146  time  0.5395410060882568
Epoch  147  time  0.5299687385559082
Epoch  148  time  0.542180061340332
Epoch  149  time  0.5342409610748291
Epoch  150  time  0.5518012046813965
Epoch  150  loss  1.3091471855588828 correct 49
Epoch  151  time  0.5235130786895752
Epoch  152  time  0.5604112148284912
Epoch  153  time  0.553861141204834
Epoch  154  time  0.5324122905731201
Epoch  155  time  0.5376880168914795
Epoch  156  time  0.5639429092407227
Epoch  157  time  0.5457842350006104
Epoch  158  time  0.537999153137207
Epoch  159  time  0.5280649662017822
Epoch  160  time  0.5467197895050049
Epoch  160  loss  0.45239326036864785 correct 50
Epoch  161  time  0.5434529781341553
Epoch  162  time  0.5190339088439941
Epoch  163  time  0.5266129970550537
Epoch  164  time  0.5329370498657227
Epoch  165  time  0.5343098640441895
Epoch  166  time  0.5254158973693848
Epoch  167  time  0.5303969383239746
Epoch  168  time  0.5488779544830322
Epoch  169  time  0.5424299240112305
Epoch  170  time  0.5758929252624512
Epoch  170  loss  0.6664771668864168 correct 50
Epoch  171  time  0.5260846614837646
Epoch  172  time  0.5550999641418457
Epoch  173  time  0.5327701568603516
Epoch  174  time  0.5458297729492188
Epoch  175  time  0.5507609844207764
Epoch  176  time  0.5409359931945801
Epoch  177  time  0.5445809364318848
Epoch  178  time  0.5335192680358887
Epoch  179  time  0.5514121055603027
Epoch  180  time  0.55706787109375
Epoch  180  loss  1.319947441272275 correct 50
Epoch  181  time  0.535768985748291
Epoch  182  time  0.5291440486907959
Epoch  183  time  0.5370481014251709
Epoch  184  time  0.5340657234191895
Epoch  185  time  0.5615749359130859
Epoch  186  time  0.554434061050415
Epoch  187  time  0.5329551696777344
Epoch  188  time  0.5714430809020996
Epoch  189  time  0.542694091796875
Epoch  190  time  0.607086181640625
Epoch  190  loss  1.025493103934656 correct 49
Epoch  191  time  0.5488231182098389
Epoch  192  time  0.5439159870147705
Epoch  193  time  0.527576208114624
Epoch  194  time  0.522191047668457
Epoch  195  time  0.5197250843048096
Epoch  196  time  0.5345909595489502
Epoch  197  time  0.5381338596343994
Epoch  198  time  0.5215029716491699
Epoch  199  time  0.5270881652832031
Epoch  200  time  0.5289478302001953
Epoch  200  loss  1.9328845225874873 correct 47
Epoch  201  time  0.5290889739990234
Epoch  202  time  0.5288589000701904
Epoch  203  time  0.5351572036743164
Epoch  204  time  0.5389587879180908
Epoch  205  time  0.5215640068054199
Epoch  206  time  0.5230100154876709
Epoch  207  time  0.5270199775695801
Epoch  208  time  0.5223221778869629
Epoch  209  time  0.52335524559021
Epoch  210  time  0.5336267948150635
Epoch  210  loss  0.2527525280971491 correct 47
Epoch  211  time  0.5283839702606201
Epoch  212  time  0.5475771427154541
Epoch  213  time  0.5316767692565918
Epoch  214  time  0.5262789726257324
Epoch  215  time  0.5611119270324707
Epoch  216  time  0.5906031131744385
Epoch  217  time  0.5449421405792236
Epoch  218  time  0.5389800071716309
Epoch  219  time  0.571497917175293
Epoch  220  time  0.562161922454834
Epoch  220  loss  1.127950673458677 correct 50
Epoch  221  time  0.5659060478210449
Epoch  222  time  0.5408790111541748
Epoch  223  time  0.5403859615325928
Epoch  224  time  0.5281989574432373
Epoch  225  time  0.5193071365356445
Epoch  226  time  0.5235841274261475
Epoch  227  time  0.5318131446838379
Epoch  228  time  0.5334160327911377
Epoch  229  time  0.5196096897125244
Epoch  230  time  0.5660820007324219
Epoch  230  loss  0.13202418495169127 correct 50
Epoch  231  time  0.5349569320678711
Epoch  232  time  0.5313460826873779
Epoch  233  time  0.5176808834075928
Epoch  234  time  0.5304279327392578
Epoch  235  time  0.5286867618560791
Epoch  236  time  0.526928186416626
Epoch  237  time  0.5196220874786377
Epoch  238  time  0.5368568897247314
Epoch  239  time  0.5336382389068604
Epoch  240  time  0.5271270275115967
Epoch  240  loss  0.020675950302090226 correct 49
Epoch  241  time  0.5462126731872559
Epoch  242  time  0.5217840671539307
Epoch  243  time  0.5211529731750488
Epoch  244  time  0.518096923828125
Epoch  245  time  0.5225710868835449
Epoch  246  time  0.5252149105072021
Epoch  247  time  0.5305521488189697
Epoch  248  time  0.5427999496459961
Epoch  249  time  0.5390799045562744
Epoch  250  time  0.5124852657318115
Epoch  250  loss  1.1269253898358462 correct 50
Epoch  251  time  0.5275518894195557
Epoch  252  time  0.5247342586517334
Epoch  253  time  0.5244741439819336
Epoch  254  time  0.5239722728729248
Epoch  255  time  0.5251209735870361
Epoch  256  time  0.5228719711303711
Epoch  257  time  0.5231778621673584
Epoch  258  time  0.5184750556945801
Epoch  259  time  0.5208659172058105
Epoch  260  time  0.5332798957824707
Epoch  260  loss  0.620780438213436 correct 47
Epoch  261  time  0.5364537239074707
Epoch  262  time  0.517582893371582
Epoch  263  time  0.5209169387817383
Epoch  264  time  0.527493953704834
Epoch  265  time  0.5215632915496826
Epoch  266  time  0.5173859596252441
Epoch  267  time  0.5351300239562988
Epoch  268  time  0.5139641761779785
Epoch  269  time  0.5281260013580322
Epoch  270  time  0.525698184967041
Epoch  270  loss  1.517082016694113 correct 50
Epoch  271  time  0.5325119495391846
Epoch  272  time  0.525554895401001
Epoch  273  time  0.5237748622894287
Epoch  274  time  0.5223050117492676
Epoch  275  time  0.5182349681854248
Epoch  276  time  0.5300891399383545
Epoch  277  time  0.5363099575042725
Epoch  278  time  0.5305449962615967
Epoch  279  time  0.5307509899139404
Epoch  280  time  0.5471160411834717
Epoch  280  loss  0.30621933431712783 correct 47
Epoch  281  time  0.5272848606109619
Epoch  282  time  0.5865058898925781
Epoch  283  time  0.5409021377563477
Epoch  284  time  0.5483429431915283
Epoch  285  time  0.5292298793792725
Epoch  286  time  0.5745992660522461
Epoch  287  time  0.5968730449676514
Epoch  288  time  0.5346629619598389
Epoch  289  time  0.5209581851959229
Epoch  290  time  0.5314350128173828
Epoch  290  loss  1.2508206492614646 correct 50
Epoch  291  time  0.5167529582977295
Epoch  292  time  0.5158209800720215
Epoch  293  time  0.5349669456481934
Epoch  294  time  0.5194940567016602
Epoch  295  time  0.5078577995300293
Epoch  296  time  0.512103796005249
Epoch  297  time  0.5318717956542969
Epoch  298  time  0.5167069435119629
Epoch  299  time  0.5229899883270264
Epoch  300  time  0.5148649215698242
Epoch  300  loss  1.7233696641852405 correct 48
Epoch  301  time  0.5278449058532715
Epoch  302  time  0.5181150436401367
Epoch  303  time  0.5275330543518066
Epoch  304  time  0.5385479927062988
Epoch  305  time  0.5356967449188232
Epoch  306  time  0.5257692337036133
Epoch  307  time  0.5170650482177734
Epoch  308  time  0.5347089767456055
Epoch  309  time  0.5109860897064209
Epoch  310  time  0.5202257633209229
Epoch  310  loss  0.899331407306326 correct 50
Epoch  311  time  0.5195698738098145
Epoch  312  time  0.5258567333221436
Epoch  313  time  0.5274231433868408
Epoch  314  time  0.7914330959320068
Epoch  315  time  0.5394008159637451
Epoch  316  time  0.551163911819458
Epoch  317  time  0.5564126968383789
Epoch  318  time  0.5825250148773193
Epoch  319  time  0.5733728408813477
Epoch  320  time  0.5386168956756592
Epoch  320  loss  7.963430156211391e-05 correct 49
Epoch  321  time  0.5204708576202393
Epoch  322  time  0.539201021194458
Epoch  323  time  0.5556950569152832
Epoch  324  time  0.5287280082702637
Epoch  325  time  0.5359227657318115
Epoch  326  time  0.5340762138366699
Epoch  327  time  0.5214247703552246
Epoch  328  time  0.5265011787414551
Epoch  329  time  0.5202088356018066
Epoch  330  time  0.5244441032409668
Epoch  330  loss  0.7133126995158781 correct 49
Epoch  331  time  0.5271120071411133
Epoch  332  time  0.5258228778839111
Epoch  333  time  0.5182371139526367
Epoch  334  time  0.5282566547393799
Epoch  335  time  0.5182368755340576
Epoch  336  time  0.5220599174499512
Epoch  337  time  0.5316972732543945
Epoch  338  time  0.536442756652832
Epoch  339  time  0.5242950916290283
Epoch  340  time  0.5200071334838867
Epoch  340  loss  1.0423856808776821 correct 50
Epoch  341  time  0.5587499141693115
Epoch  342  time  0.5173606872558594
Epoch  343  time  0.5594902038574219
Epoch  344  time  0.5161902904510498
Epoch  345  time  0.5243487358093262
Epoch  346  time  0.5282549858093262
Epoch  347  time  0.5456361770629883
Epoch  348  time  0.5193829536437988
Epoch  349  time  0.516542911529541
Epoch  350  time  0.5116291046142578
Epoch  350  loss  0.7476387164010565 correct 49
Epoch  351  time  0.5269389152526855
Epoch  352  time  0.526932954788208
Epoch  353  time  0.5275061130523682
Epoch  354  time  0.5390579700469971
Epoch  355  time  0.5380349159240723
Epoch  356  time  0.529534101486206
Epoch  357  time  0.5369961261749268
Epoch  358  time  0.5633499622344971
Epoch  359  time  0.5423548221588135
Epoch  360  time  0.5399610996246338
Epoch  360  loss  0.7524936385241828 correct 49
Epoch  361  time  0.5572118759155273
Epoch  362  time  0.52571702003479
Epoch  363  time  0.5252690315246582
Epoch  364  time  0.5168201923370361
Epoch  365  time  0.5304830074310303
Epoch  366  time  0.5209190845489502
Epoch  367  time  0.5539448261260986
Epoch  368  time  0.5506491661071777
Epoch  369  time  0.5576567649841309
Epoch  370  time  0.5618720054626465
Epoch  370  loss  0.1813390946621127 correct 49
Epoch  371  time  0.5608108043670654
Epoch  372  time  0.5527138710021973
Epoch  373  time  0.5480079650878906
Epoch  374  time  0.5505719184875488
Epoch  375  time  0.5233972072601318
Epoch  376  time  0.5346357822418213
Epoch  377  time  0.5178289413452148
Epoch  378  time  0.5298869609832764
Epoch  379  time  0.5263538360595703
Epoch  380  time  0.5358142852783203
Epoch  380  loss  1.299864565627423 correct 50
Epoch  381  time  0.5478231906890869
Epoch  382  time  0.5484187602996826
Epoch  383  time  0.5170402526855469
Epoch  384  time  0.5396339893341064
Epoch  385  time  0.5143091678619385
Epoch  386  time  0.5240330696105957
Epoch  387  time  0.5153119564056396
Epoch  388  time  0.5244889259338379
Epoch  389  time  0.5342981815338135
Epoch  390  time  0.5577831268310547
Epoch  390  loss  1.3934226171449193 correct 48
Epoch  391  time  0.5227560997009277
Epoch  392  time  0.5251598358154297
Epoch  393  time  0.5229701995849609
Epoch  394  time  0.53141188621521
Epoch  395  time  0.5564391613006592
Epoch  396  time  0.5600991249084473
Epoch  397  time  0.5461888313293457
Epoch  398  time  0.5340068340301514
Epoch  399  time  0.5162029266357422
Epoch  400  time  0.5192723274230957
Epoch  400  loss  0.9568757341646795 correct 50
Epoch  401  time  0.5199441909790039
Epoch  402  time  0.5245110988616943
Epoch  403  time  0.5197560787200928
Epoch  404  time  0.5374422073364258
Epoch  405  time  0.5221359729766846
Epoch  406  time  0.522183895111084
Epoch  407  time  0.5502538681030273
Epoch  408  time  0.554157018661499
Epoch  409  time  0.5772879123687744
Epoch  410  time  0.5559060573577881
Epoch  410  loss  1.3103969913539677 correct 47
Epoch  411  time  0.5493178367614746
Epoch  412  time  0.5350008010864258
Epoch  413  time  0.5409691333770752
Epoch  414  time  0.5259947776794434
Epoch  415  time  0.5302383899688721
Epoch  416  time  0.5171756744384766
Epoch  417  time  0.5249452590942383
Epoch  418  time  0.514171838760376
Epoch  419  time  0.5365219116210938
Epoch  420  time  0.5161011219024658
Epoch  420  loss  0.002291794323642712 correct 50
Epoch  421  time  0.5191192626953125
Epoch  422  time  0.519913911819458
Epoch  423  time  0.5210649967193604
Epoch  424  time  0.5297322273254395
Epoch  425  time  0.5321879386901855
Epoch  426  time  0.525205135345459
Epoch  427  time  0.5213570594787598
Epoch  428  time  0.530203104019165
Epoch  429  time  0.5312409400939941
Epoch  430  time  0.5963408946990967
Epoch  430  loss  0.028015887452383894 correct 49
Epoch  431  time  0.5156421661376953
Epoch  432  time  0.5239119529724121
Epoch  433  time  0.5326390266418457
Epoch  434  time  0.5228531360626221
Epoch  435  time  0.5304410457611084
Epoch  436  time  0.5075089931488037
Epoch  437  time  0.5216271877288818
Epoch  438  time  0.5152468681335449
Epoch  439  time  0.5267441272735596
Epoch  440  time  0.5318882465362549
Epoch  440  loss  1.1771689065412727 correct 49
Epoch  441  time  0.5504148006439209
Epoch  442  time  0.5528719425201416
Epoch  443  time  0.5775630474090576
Epoch  444  time  0.5320479869842529
Epoch  445  time  0.5256779193878174
Epoch  446  time  0.5307440757751465
Epoch  447  time  0.523015022277832
Epoch  448  time  0.5238029956817627
Epoch  449  time  0.5241918563842773
Epoch  450  time  0.5193531513214111
Epoch  450  loss  1.1585067458150553 correct 48
Epoch  451  time  0.5431652069091797
Epoch  452  time  0.5545930862426758
Epoch  453  time  0.538222074508667
Epoch  454  time  0.5375790596008301
Epoch  455  time  0.5209088325500488
Epoch  456  time  0.5299558639526367
Epoch  457  time  0.5226001739501953
Epoch  458  time  0.5249819755554199
Epoch  459  time  0.522568941116333
Epoch  460  time  0.5382900238037109
Epoch  460  loss  0.8697354256788162 correct 50
Epoch  461  time  0.5195660591125488
Epoch  462  time  0.5227148532867432
Epoch  463  time  0.5158991813659668
Epoch  464  time  0.5531589984893799
Epoch  465  time  0.5256967544555664
Epoch  466  time  0.5520198345184326
Epoch  467  time  0.5597600936889648
Epoch  468  time  0.5209341049194336
Epoch  469  time  0.5206680297851562
Epoch  470  time  0.5094699859619141
Epoch  470  loss  0.49543193114685646 correct 50
Epoch  471  time  0.5301051139831543
Epoch  472  time  0.5344350337982178
Epoch  473  time  0.516505241394043
Epoch  474  time  0.5349791049957275
Epoch  475  time  0.5615150928497314
Epoch  476  time  0.5254521369934082
Epoch  477  time  0.5164089202880859
Epoch  478  time  0.5229618549346924
Epoch  479  time  0.5343010425567627
Epoch  480  time  0.519989013671875
Epoch  480  loss  1.1303771909227267 correct 50
Epoch  481  time  0.5158882141113281
Epoch  482  time  0.5539169311523438
Epoch  483  time  0.5425398349761963
Epoch  484  time  0.579279899597168
Epoch  485  time  0.5189192295074463
Epoch  486  time  0.541632890701294
Epoch  487  time  0.531984806060791
Epoch  488  time  0.5425169467926025
Epoch  489  time  0.5229091644287109
Epoch  490  time  0.5208420753479004
Epoch  490  loss  0.6147665799799507 correct 50
Epoch  491  time  0.5244390964508057
Epoch  492  time  0.5240738391876221
Epoch  493  time  0.5209300518035889
Epoch  494  time  0.5362100601196289
Epoch  495  time  0.5260858535766602
Epoch  496  time  0.5230529308319092
Epoch  497  time  0.5266060829162598
Epoch  498  time  0.5260219573974609
Epoch  499  time  0.5620269775390625
</details>