package weka.python;

import ai.rapids.cudf.HostBufferConsumer;
import ai.rapids.cudf.HostMemoryBuffer;

import java.io.*;

public class TableBuffer implements HostBufferConsumer {
    ByteArrayOutputStream tableBuffer = new ByteArrayOutputStream();

    /**
     * Consume a buffer.
     *
     * @param buffer the buffer.  Be sure to close this buffer when you are done
     *               with it or it will leak.
     * @param len    the length of the buffer that is valid.  The valid data will be 0 until len.
     */
    @Override
    public void handleBuffer(HostMemoryBuffer buffer, long len) {
        for (long i = 0; i < len; i++) {
            byte b = buffer.getByte(i);
            tableBuffer.write(b);
        }
        buffer.close();
    }

    public void writeToFile(File file) {
        try (OutputStream outputStream = new FileOutputStream(file)) {
            tableBuffer.writeTo(outputStream);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public byte[] toByteArray() {
        return tableBuffer.toByteArray();
    }
}
