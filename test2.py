import pyopencl as cl
import numpy
import numpy.linalg as la

block_size = 16


class NaiveTranspose:
    def __init__(self, ctx):
        self.kernel = cl.Program(ctx, """
        __kernel
        void transpose(
          __global float *a_t, __global float *a,
          unsigned a_width, unsigned a_height)
        {
          int read_idx = get_global_id(0) + get_global_id(1) * a_width;
          int write_idx = get_global_id(1) + get_global_id(0) * a_height;
          a_t[write_idx] = a[read_idx];
        }
        """ % {"block_size": block_size}).build().transpose

    def __call__(self, queue, tgt, src, shape):
        w, h = shape
        assert w % block_size == 0
        assert h % block_size == 0

        return self.kernel(queue, (w, h), (block_size, block_size),
            tgt, src, numpy.uint32(w), numpy.uint32(h))

ctx = cl.create_some_context()
for dev in ctx.devices:
    assert dev.local_mem_size > 0

queue = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)
source = numpy.random.rand(1024, 1024).astype(numpy.float32)

mf = cl.mem_flags
a_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=source)
a_t_buf = cl.Buffer(ctx, mf.WRITE_ONLY, size=source.nbytes)

method = NaiveTranspose(ctx)

print("TESTS:")
a = method(queue, a_t_buf, a_buf, source.shape)
b = numpy.empty_like(source)
c = numpy.empty_like(source)
cl.enqueue_copy(queue, b, a_t_buf)
cl.enqueue_copy(queue, c, a_buf)
print(b)
print(c)
