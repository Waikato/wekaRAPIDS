package weka.python;

import org.bytedeco.javacpp.FloatPointer;
import weka.core.Instances;
import weka.core.WekaException;

public class InstanceSender {
    /**
     * instance sending method: CSV
     */
    public static final int INSTANCE_SENDING_CSV = 0;
    /**
     * instance sending method: Arrow IPC
     */
    public static final int INSTANCE_SENDING_ARROW_IPC = 1;
    /**
     * instance sending method: Shared GPU Memory
     */
    public static final int INSTANCE_SENDING_SHARED_GPU_MEMORY = 2;
    private final boolean m_Debug;

    /**
     * The current method sending instances
     */
    protected int m_SendingMethod;

    private FloatPointer m_DataPtr = null;

    private final RapidsSession m_Session;

    public InstanceSender(RapidsSession session, int method, boolean debug) {
        m_Session = session;
        m_SendingMethod = method;
        m_Debug = debug;
    }

    public void sendInstances(Instances instances, String pythonFrameName, boolean debug) throws WekaException {
        reset();
        if (m_Debug) {
            System.out.println("Sending instances via method: " + m_SendingMethod);
        }

        //Start time
        long begin = System.currentTimeMillis();

        switch (m_SendingMethod) {
            case INSTANCE_SENDING_CSV:
                m_Session.instancesToPythonAsScikitLearn(instances, pythonFrameName, debug);
                break;
            case INSTANCE_SENDING_SHARED_GPU_MEMORY:
                m_DataPtr = new FloatPointer();
                m_Session.instancesToPythonAsCuda(instances, m_DataPtr, pythonFrameName, debug);
                break;
            default: // INSTANCE_SENDING_ARROW_IPC
                m_Session.instancesToPythonAsArrow(instances, pythonFrameName, debug);
        }

        //End time
        long end = System.currentTimeMillis();

        if (m_Debug) {
            System.out.println("Time taken to send instances: " + (end - begin) + "ms");
        }
    }

    @Override
    protected void finalize() throws WekaException {
        reset();
    }

    protected void reset() throws WekaException {
        if (m_DataPtr != null) {
            m_Session.closeIPCHandle(m_Debug);
            m_Session.freeMemory(m_DataPtr);
            m_DataPtr = null;
        }
    }

    public void release() throws WekaException {
        reset();
    }
}
