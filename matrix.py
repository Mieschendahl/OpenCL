import pyopencl as cl
import numpy as np

block_size = 2
ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

class AddMat:
    def __init__(self, ctx, queue):
        code = """
        __kernel void add(unsigned dimsize, __global const unsigned *dim, __global const float *a, __global const float *b, __global float *c) {
            unsigned gid = 0;
            unsigned acc = 1;
            for (int i = 0; i < dimsize; i++) {
                gid += get_global_id(i) * acc;
                acc *= dim[i];
            }

            c[gid] = a[gid] + b[gid];
        }
        """
        self.kernel = cl.Program(ctx, code).build().add
        self.queue = queue

    def __call__(self, a, b):
        assert a.shape == b.shape, "Matricies should have the same dimension!"
        assert all(not(x % block_size) for x in a.shape), "Matrix dimensions are not multiples of the block size (%s)!" % block_size
        dim = cl.Buffer(ctx, cl.mem_flags.READ_ONLY |cl.mem_flags.COPY_HOST_PTR, hostbuf=np.array(a.shape, dtype=np.uint32))
        a_g = cl.Buffer(ctx, cl.mem_flags.READ_ONLY |cl.mem_flags.COPY_HOST_PTR, hostbuf=np.array(a, dtype=np.float32))
        b_g = cl.Buffer(ctx, cl.mem_flags.READ_ONLY |cl.mem_flags.COPY_HOST_PTR, hostbuf=np.array(b, dtype=np.float32))
        c_g = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, size=np.array(a, dtype=np.float32).nbytes)

        self.kernel(self.queue, a.shape, tuple([block_size] * len(a.shape)), np.uint32(len(a.shape)), dim, a_g, b_g, c_g)
        c = np.empty_like(a, dtype=np.float32)
        cl.enqueue_copy(queue, c, c_g)
        return c

a = np.array([[1, 2,], [3, 4]])
b = np.array([[2, 3], [5, 6]])

addmat = AddMat(ctx, queue)
c = addmat(a, b)

print("A:\n", a, sep="")
print("B:\n", b, sep="")
print("C:\n", c, sep="")
