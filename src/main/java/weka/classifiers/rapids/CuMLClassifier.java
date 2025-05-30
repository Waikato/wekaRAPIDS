/*
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

package weka.classifiers.rapids;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.rules.ZeroR;
import weka.core.*;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;
import weka.python.InstanceSender;
import weka.python.RapidsSession;

import java.util.Collections;
import java.util.List;

import static weka.python.InstanceSender.*;

/**
 * Wrapper classifier for classifiers and regressors implemented in the
 * cuML Python package.
 *
 * @author Justin Liu (justin{[dot]}l{[at]}waikato{[dot]}ac{[dot]}nz)
 * @version $Revision: $
 */
public class CuMLClassifier extends AbstractClassifier
        implements BatchPredictor, CapabilitiesHandler {

    protected static final String TRAINING_DATA_ID = "rapids_classifier_training";
    protected static final String TEST_DATA_ID = "cuml_classifier_test";
    protected static final String MODEL_ID = "weka_cuml_learner";

    /**
     * For serialization
     */
    private static final long serialVersionUID = -8054156420780803790L;

    /**
     * instance sending method
     */
    public static final Tag[] TAGS_INSTANCE_SENDING = {
            new Tag(INSTANCE_SENDING_CSV, "Send instances by CSV format"),
            new Tag(INSTANCE_SENDING_ARROW_IPC, "Send Instances by ARROW IPC"),
            new Tag(INSTANCE_SENDING_SHARED_GPU_MEMORY, "Shared Instances in GPU Memory")};

    /**
     * The current method to send instances
     */
    protected int m_sendingMethod = 1;

    /**
     * Holds info on the different learners available
     */
    public static enum Learner {
        LinearRegression("linear_model", false, true, false, false,
                "\talgorithm='eig', fit_intercept=True, normalize=False, handle=None, verbose=False"),
        LogisticRegression("linear_model", true, false, true, false,
                "\tpenalty='l2', tol=0.0001, C=1.0, fit_intercept=True, class_weight=None,\n"
                        + "\tmax_iter=1000, linesearch_max_iter=50, verbose=False, l1_ratio=None, solver='qn',\n"
                        + "handle=None, output_type=None"),
        Ridge("linear_model", false, true, false, false,
                "\talpha=1.0, solver='eig', fit_intercept=True, normalize=False, handle=None,\n"
                        + "\toutput_type=None, verbose=False"),
        Lasso("linear_model", false, true, false, false,
                "\talpha=1.0, fit_intercept=True, normalize=False, max_iter=1000, tol=0.001,\n"
                        + "\t selection='cyclic', handle=None, output_type=None, verbose=False"),
        ElasticNet("linear_model", false, true, false, false,
                "\talpha=1.0, l1_ratio=0.5, fit_intercept=True, normalize=False, max_iter=1000,\n"
                        + "\ttol=0.001, selection='cyclic', handle=None, output_type=None, verbose=False"),
        MBSGDClassifier("linear_model", true, false, false, false,
                "\tloss='hinge', penalty='l2', alpha=0.0001, l1_ratio=0.15, fit_intercept=True,\n"
                        + "\tepochs=1000, tol=0.001, shuffle=True, learning_rate='constant', eta0=0.001,\n"
                        + "\tpower_t=0.5, batch_size=32, n_iter_no_change=5, handle=None, verbose=False,\n"
                        + "\toutput_type=None"),
        MBSGDRegressor("linear_model", false, true, false, false,
                "\tloss='squared_loss', penalty='l2', alpha=0.0001, l1_ratio=0.15, fit_intercept=True,\n"
                        + "\tepochs=1000, tol=0.001, shuffle=True, learning_rate='constant', eta0=0.001,\n"
                        + "\tpower_t=0.5, batch_size=32, n_iter_no_change=5, handle=None, verbose=False,\n"
                        + "\toutput_type=None"),
        MultinomialNB("naive_bayes", true, false, true, false,
                "\talpha=1.0, fit_prior=True, class_prior=None, output_type=None, handle=None, verbose=False"),
        BernoulliNB("naive_bayes", true, false, true, false,
                "\talpha=1.0, binarize=0.0, fit_prior=True, class_prior=None, output_type=None, handle=None, verbose=False"),
        GaussianNB("naive_bayes", true, false, true, false,
                "\tpriors=None, var_smoothing=1e-09, output_type=None, handle=None, verbose=False"),
        RandomForestClassifier("ensemble", true, false, true, false,
                "\tn_estimators=100, split_criterion=0, bootstrap=True, max_samples=1.0, max_depth=16,\n"
                        + "\tmax_leaves=-1, max_features='auto', n_bins=128, n_streams=4, min_samples_leaf=1,\n"
                        + "\tmin_samples_split=2, min_impurity_decrease=0.0, max_batch_size=4096, random_state=None,\n"
                        + "\thandle=None, verbose=False, output_type=None"),
        RandomForestRegressor("ensemble", false, true, false, false,
                "\tn_estimators=100, split_criterion=0, bootstrap=True, max_samples=1.0, max_depth=16,\n"
                        + "\tmax_leaves=-1, max_features='auto', n_bins=128, n_streams=4, min_samples_leaf=1,\n"
                        + "\tmin_samples_split=2, min_impurity_decrease=0.0, accuracy_metric='r2', max_batch_size=4096,\n"
                        + "\trandom_state=None, handle=None, verbose=False, output_type=None"),
        SVC("svm", true, false, false, false,
                "\thandle=None, C=1.0, kernel='rbf', degree=3, gamma='scale', coef0=0.0, tol=1e-3,\n"
                        + "\tcache_size=1024.0, class_weight=None, multiclass_strategy='ovo', nochange_steps=1000,\n"
                        + "\toutput_type=None, probability=False, random_state=None, verbose=False"),
        SVR("svm", false, true, false, false,
                "\thandle=None, C=1.0, kernel='rbf', degree=3, gamma='scale', coef0=0.0, tol=1e-3,\n"
                        + "\tepsilon=0.1, cache_size=1024.0, nochange_steps=1000, verbose=False, output_type=None"),
        LinearSVC("svm", true, false, false, false,
                "\thandle=None, penalty='l2', loss='squared_hinge', fit_intercept=True,\n"
                        + "\tpenalized_intercept=False, max_iter=1000, linesearch_max_iter=1000, lbfgs_memory=5,\n"
                        + "\tverbose=False, C=1.0, grad_tol=0.0001, change_tol=1e-05, tol=None, probability=False,\n"
                        + "\tmulti_class=False, output_type=None"),
        KNeighborsClassifier("neighbors", true, false, true, false,
                "n_neighbors=5, algorithm='brute', metric='euclidean', weights='uniform', handle=None,\n"
                        + "\tverbose=False, output_type=None"),
        KNeighborsRegressor("neighbors", false, true, false, false,
                "n_neighbors=5, algorithm='brute', metric='euclidean', weights='uniform', handle=None,\n"
                        + "\tverbose=False, output_type=None");

        private String m_module;
        private boolean m_classification;
        private boolean m_regression;
        private boolean m_producesProbabilities;

        /**
         * True to have the model variable cleared in python after training and
         * batch prediction. This conserves memory, and might allow additional
         * off-heap resources (e.g. GPU device memory) to be freed. However, there
         * will be additional overhead in transferring the model back into python
         * each time predictions are required.
         */
        private boolean m_removeModelFromPyPostTraining;
        private String m_defaultParameters;

        /**
         * Enum constructor
         *
         * @param module                the cuML module of the given scheme
         * @param classification        true if it is a classifier
         * @param regression            true if it is a regressor
         * @param producesProbabilities true if it produces probabilities
         * @param defaultParameters     the list of default parameter settings
         */
        Learner(String module, boolean classification, boolean regression,
                boolean producesProbabilities, boolean removeModel,
                String defaultParameters) {
            m_module = module;
            m_producesProbabilities = producesProbabilities;
            m_classification = classification;
            m_regression = regression;
            m_removeModelFromPyPostTraining = removeModel;
            m_defaultParameters = defaultParameters;
        }

        /**
         * Get the cuML module of this scheme
         *
         * @return the cuML module of this scheme
         */
        public String getModule() {
            return m_module;
        }

        /**
         * Default implementation of producesProbabilities given parameter settings.
         * Specific enum values can override if there are specific parameter
         * settings where probabilities can or can't be produced
         *
         * @param params the current parameter settings for the scheme
         * @return true if probabilities can be produced given the parameter
         * settings
         */
        public boolean producesProbabilities(String params) {
            return m_producesProbabilities;
        }

        /**
         * Return true if the variable containing the model in python should be
         * cleared after training and each batch prediction operation. This can be
         * advantageous for specific methods that might either consume lots of host
         * memory or device (e.g. GPU) memory. The disadvantage is that the model
         * will need to be transferred into python prior to each batch prediction
         * call (cross-validation will be slower).
         *
         * @return true if the model variable should be cleared in python
         */
        public boolean removeModelFromPythonPostTrainPredict() {
            return m_removeModelFromPyPostTraining;
        }

        /**
         * Return true if this scheme is a classifier
         *
         * @return true if this scheme is a classifier
         */
        public boolean isClassifier() {
            return m_classification;
        }

        /**
         * Return true if this scheme is a regressor
         *
         * @return true if this scheme is a regressor
         */
        public boolean isRegressor() {
            return m_regression;
        }

        /**
         * Get the default settings for parameters for this scheme
         *
         * @return the default parameter settings of this scheme
         */
        public String getDefaultParameters() {
            return m_defaultParameters;
        }
    }

    ;

    /**
     * The tags for the GUI drop-down for learner selection
     */
    public static final Tag[] TAGS_LEARNER = new Tag[Learner.values().length];

    static {
        for (Learner l : Learner.values()) {
            TAGS_LEARNER[l.ordinal()] = new Tag(l.ordinal(), l.toString());
        }
    }

    /**
     * Holds the version number of cuML.
     */
    protected double m_cumlVersion = -1;

    /**
     * The cuml learner to use
     */
    protected Learner m_learner = Learner.RandomForestClassifier;

    /**
     * The parameters to pass to the learner
     */
    protected String m_learnerOpts = "";

    /**
     * True if the supervised nominal to binary filter should be used
     */
    protected boolean m_useSupervisedNominalToBinary;

    /**
     * The nominal to binary filter
     */
    protected Filter m_nominalToBinary;

    /**
     * For replacing missing values
     */
    protected Filter m_replaceMissing = new ReplaceMissingValues();

    /**
     * Holds the textual description of the cuml learner
     */
    protected String m_learnerToString = "";

    /**
     * For making this model unique in python
     */
    protected String m_modelHash;

    /**
     * True for nominal class labels that don't occur in the training data
     */
    protected boolean[] m_nominalEmptyClassIndexes;

    /**
     * Fall back to Zero R if there are no instances with non-missing class or
     * only the class is present in the data
     */
    protected ZeroR m_zeroR;

    /**
     * Class priors for use if there are numerical problems
     */
    protected double[] m_classPriors;

    /**
     * Default - use python in PATH
     */
    protected String m_pyCommand = "default";

    /**
     * Default - use PATH as is for executing python
     */
    protected String m_pyPath = "default";

    /**
     * Optional server name/ID by which to force a new server for this instance
     * (or share amongst specific instances). The string "none" indicates no
     * specific server name. Python command + server name uniquely identifies a
     * server to use
     */
    protected String m_serverID = "none";

    /**
     * Global help info
     *
     * @return the global help info for this scheme
     */
    public String globalInfo() {
        StringBuilder b = new StringBuilder();
        b.append("A wrapper for classifiers implemented in the cuML "
                + "python library. The following learners are available:\n\n");
        for (Learner l : Learner.values()) {
            b.append(l.toString()).append("\n");
            b.append("[");
            if (l.isClassifier()) {
                b.append(" classification ");
            }
            if (l.isRegressor()) {
                b.append(" regression ");
            }
            b.append("]").append("\nDefault parameters:\n");
            b.append(l.getDefaultParameters()).append("\n");
        }
        return b.toString();
    }

    /**
     * Batch prediction size (default: 100 instances)
     */
    protected String m_batchPredictSize = "100";

    /**
     * Whether to try and continue after script execution reports output on system
     * error from Python. Some schemes output warning (not error) messages to the
     * sys err stream
     */
    protected boolean m_continueOnSysErr;

    /**
     * Get the capabilities of this learner
     *
     * @param doServerChecks true if the python server checks should be run
     * @return the Capabilities of this learner
     */
    protected Capabilities getCapabilities(boolean doServerChecks) {
        Capabilities result = super.getCapabilities();
        result.disableAll();

        if (doServerChecks) {
            boolean pythonAvailable = true;
            RapidsSession session = null;
            try {
                session = getSession();
            } catch (WekaException e) {
                pythonAvailable = false;
            }

            if (pythonAvailable) {
                if (m_cumlVersion < 0) {
                    // try and establish cuML version
                    try {
                        // PythonSession session = PythonSession.acquireSession(this);
                        String script = "import cuml\ncumlv = cuml.__version__\n";
                        List<String> outAndErr = session.executeScript(script, getDebug());

                        String versionNumber =
                                session.getVariableValueFromPythonAsPlainString("cumlv", getDebug());
                        if (versionNumber != null && versionNumber.length() > 0) {
                            // strip minor version
                            versionNumber = versionNumber.substring(0, versionNumber.lastIndexOf('.'));
                            m_cumlVersion = Double.parseDouble(versionNumber);
                        }
                    } catch (WekaException e) {
                        e.printStackTrace();
                    } finally {
                        try {
                            releaseSession();
                            session = null;
                        } catch (WekaException e) {
                            e.printStackTrace();
                        }
                    }
                }

                result.enable(Capabilities.Capability.NUMERIC_ATTRIBUTES);
                result.enable(Capabilities.Capability.NOMINAL_ATTRIBUTES);
                result.enable(Capabilities.Capability.DATE_ATTRIBUTES);
                result.enable(Capabilities.Capability.MISSING_VALUES);
                result.enable(Capabilities.Capability.MISSING_CLASS_VALUES);
                if (m_learner.isClassifier()) {
                    result.enable(Capabilities.Capability.BINARY_CLASS);
                    result.enable(Capabilities.Capability.NOMINAL_CLASS);
                }
                if (m_learner.isRegressor()) {
                    result.enable(Capabilities.Capability.NUMERIC_CLASS);
                }
                if (session != null) {
                    try {
                        releaseSession();
                    } catch (WekaException e) {
                        e.printStackTrace();
                    }
                }
            } else {
                try {
                    System.err.println(
                            "The python environment is either not available or " + "is not configured correctly:\n\n"
                                    + getPythonEnvCheckResults());
                } catch (WekaException e) {
                    e.printStackTrace();
                }
            }
        } else {
            result.enable(Capabilities.Capability.NUMERIC_ATTRIBUTES);
            result.enable(Capabilities.Capability.NOMINAL_ATTRIBUTES);
            result.enable(Capabilities.Capability.DATE_ATTRIBUTES);
            result.enable(Capabilities.Capability.MISSING_VALUES);
            result.enable(Capabilities.Capability.MISSING_CLASS_VALUES);
            if (m_learner.isClassifier()) {
                result.enable(Capabilities.Capability.BINARY_CLASS);
                result.enable(Capabilities.Capability.NOMINAL_CLASS);
            }
            if (m_learner.isRegressor()) {
                result.enable(Capabilities.Capability.NUMERIC_CLASS);
            }
        }

        return result;
    }

    /**
     * Get the capabilities of this learner
     *
     * @return the capabilities of this scheme
     */
    @Override
    public Capabilities getCapabilities() {
        // note that we don't do the server checks from here because the GenericObjectEditor
        // in the GUI calls this method for every character typed in a text field!
        return getCapabilities(false);
    }

    /**
     * Get whether to use the supervised version of nominal to binary
     *
     * @return true if supervised nominal to binary is to be used
     */
    @OptionMetadata(displayName = "Use supervised nominal to binary conversion",
            description = "Use supervised nominal to binary conversion of nominal attributes.",
            commandLineParamName = "S", commandLineParamSynopsis = "-S",
            commandLineParamIsFlag = true, displayOrder = 3)
    public boolean getUseSupervisedNominalToBinary() {
        return m_useSupervisedNominalToBinary;
    }

    /**
     * Set whether to use the supervised version of nominal to binary
     *
     * @param useSupervisedNominalToBinary true if supervised nominal to binary is
     *                                     to be used
     */
    public void
    setUseSupervisedNominalToBinary(boolean useSupervisedNominalToBinary) {
        m_useSupervisedNominalToBinary = useSupervisedNominalToBinary;
    }

    /**
     * Get the cuML scheme to use
     *
     * @return the cuML scheme to use
     */
    @OptionMetadata(displayName = "RAPIDS cuML learner",
            description = "RAPIDS cuML learner to use.\nAvailable learners:\n"
                    + "LinearRegression\n"
                    + "LogisticRegression\n"
                    + "Ridge\n"
                    + "Lasso\n"
                    + "ElasticNet\n"
                    + "MBSGDClassifier\n"
                    + "MBSGDRegressor\n"
                    + "MultinomialNB\n"
                    + "BernoulliNB\n"
                    + "GaussianNB\n"
                    + "RandomForestClassifier\n"
                    + "RandomForestRegression\n"
                    + "SVC\n"
                    + "SVR\n"
                    + "LinearSVC\n"
                    + "KNeighborsRegressor\n"
                    + "KNeighborsClassifier\n"
                    + "\n(default = RandomForestClassifier)",
            commandLineParamName = "learner",
            commandLineParamSynopsis = "-learner <learner name>", displayOrder = 1)
    public SelectedTag getLearner() {
        return new SelectedTag(m_learner.ordinal(), TAGS_LEARNER);
    }

    /**
     * Set the cuML scheme to use
     *
     * @param learner the cuML scheme to use
     */
    public void setLearner(SelectedTag learner) {
        int learnerID = learner.getSelectedTag().getID();
        for (Learner l : Learner.values()) {
            if (l.ordinal() == learnerID) {
                m_learner = l;
                break;
            }
        }
    }

    /**
     * Get the parameters to pass to the cuML scheme
     *
     * @return the parameters to use
     */
    @OptionMetadata(displayName = "Learner parameters",
            description = "learner parameters to use", displayOrder = 2,
            commandLineParamName = "parameters",
            commandLineParamSynopsis = "-parameters <comma-separated list of name=value pairs>")
    public String getLearnerOpts() {
        return m_learnerOpts;
    }

    /**
     * Set the parameters to pass to the cuML scheme
     *
     * @param opts the parameters to use
     */
    public void setLearnerOpts(String opts) {
        m_learnerOpts = opts;
    }

    @Override
    public void setBatchSize(String size) {
        m_batchPredictSize = size;
    }

    @OptionMetadata(displayName = "Batch size", description = "The preferred "
            + "number of instances to transfer into python for prediction\n(if operating"
            + "in batch prediction mode). More or fewer instances than this will be "
            + "accepted.", commandLineParamName = "batch",
            commandLineParamSynopsis = "-batch <batch size>", displayOrder = 4)
    @Override
    public String getBatchSize() {
        return m_batchPredictSize;
    }

    @Override
    public void setOptions(String[] options) throws Exception {
        String selectionString = Utils.getOption('M', options);
        if (selectionString.length() != 0) {
            setInstanceSendingStrategy(new SelectedTag(
                    Integer.parseInt(selectionString), TAGS_INSTANCE_SENDING));
        } else {
            setInstanceSendingStrategy(new SelectedTag(INSTANCE_SENDING_ARROW_IPC, TAGS_INSTANCE_SENDING));
        }

        super.setOptions(options);
    }

    @Override
    public String[] getOptions() {
        java.util.Vector<String> result = new java.util.Vector<String>();
        result.add("-M");
        result.add("" + m_sendingMethod);

        Collections.addAll(result, super.getOptions());
        return result.toArray(new String[result.size()]);
    }

    /**
     * Returns true, as we send entire test sets over to python for prediction
     *
     * @return true
     */
    @Override
    public boolean implementsMoreEfficientBatchPrediction() {
        return true;
    }

    /**
     * Set whether to try and continue after seeing output on the sys error
     * stream. Some schemes write warnings (rather than errors) to sys error.
     *
     * @param c true if we should try to continue after seeing output on the sys
     *          error stream
     */
    public void setContinueOnSysErr(boolean c) {
        m_continueOnSysErr = c;
    }

    /**
     * Get whether to try and continue after seeing output on the sys error
     * stream. Some schemes write warnings (rather than errors) to sys error.
     *
     * @return true if we should try to continue after seeing output on the sys
     * error stream
     */
    @OptionMetadata(
            displayName = "Try to continue after sys err output from script",
            description = "Try to continue after sys err output from script.\nSome schemes"
                    + " report warnings to the system error stream.",
            displayOrder = 5, commandLineParamName = "continue-on-err",
            commandLineParamSynopsis = "-continue-on-err",
            commandLineParamIsFlag = true)
    public boolean getContinueOnSysErr() {
        return m_continueOnSysErr;
    }

    /**
     * Set the python command to use. Empty string or "default" indicate that the
     * python present in the PATH should be used.
     *
     * @param pyCommand the path to the python executable (or empty
     *                  string/"default" to use python in the PATH)
     */
    public void setPythonCommand(String pyCommand) {
        m_pyCommand = pyCommand;
    }

    /**
     * Get the python command to use. Empty string or "default" indicate that the
     * python present in the PATH should be used.
     *
     * @return the path to the python executable (or empty string/"default" to use
     * python in the PATH)
     */
    @OptionMetadata(displayName = "Python command ",
            description = "Path to python executable ('default' to use python in the PATH)",
            commandLineParamName = "py-command",
            commandLineParamSynopsis = "-py-command <path to python executable>",
            displayOrder = 7)
    public String getPythonCommand() {
        return m_pyCommand;
    }

    /**
     * Set optional entries to prepend to the PATH so that python can execute
     * correctly. Only applies when not using a default python.
     *
     * @param pythonPath additional entries to prepend to the PATH
     */
    public void setPythonPath(String pythonPath) {
        m_pyPath = pythonPath;
    }

    /**
     * Get optional entries to prepend to the PATH so that python can execute
     * correctly. Only applies when not using a default python.
     *
     * @return additional entries to prepend to the PATH
     */
    @OptionMetadata(displayName = "Python path",
            description = "Optional elements to prepend to the PATH so that python can "
                    + "execute correctly ('default' to use PATH as-is)",
            commandLineParamName = "py-path",
            commandLineParamSynopsis = "-py-path <path>", displayOrder = 8)
    public String getPythonPath() {
        return m_pyPath;
    }

    /**
     * Set an optional server name by which to identify the python server to use.
     * Can be used share a given server amongst selected instances or reserve a
     * server specifically for this classifier. Python command + server name
     * uniquely identifies the server to use.
     *
     * @param serverID the name of the server to use (or none for no specific
     *                 server name).
     */
    public void setServerID(String serverID) {
        m_serverID = serverID;
    }

    /**
     * Get an optional server name by which to identify the python server to use.
     * Can be used share a given server amongst selected instances or reserve a
     * server specifically for this classifier. Python command + server name
     * uniquely identifies the server to use.
     *
     * @return the name of the server to use (or none) for no specific server
     * name.
     */
    @OptionMetadata(displayName = "Server name/ID",
            description = "Optional name to identify this server, can be used to share "
                    + "a given server instance - default = 'none' (i.e. no server name)",
            commandLineParamName = "server-name",
            commandLineParamSynopsis = "-server-name <server name | none>",
            displayOrder = 9)
    public String getServerID() {
        return m_serverID;
    }


    /**
     * Set the instances sending method
     */
    @OptionMetadata(displayName = "Instance sending method",
            description = "The method of sending instances from WEKA to Python server:\n"
                    + "CSV\n"
                    + "Arrow IPC\n"
                    + "Shared GPU Memory\n"
                    + "\n(default = Arrow IPC)",
            commandLineParamName = "M",
            commandLineParamSynopsis = "-M <method number>", displayOrder = 10)
    public void setInstanceSendingStrategy(SelectedTag method) {
        if (method.getTags() == TAGS_INSTANCE_SENDING) {
            m_sendingMethod = method.getSelectedTag().getID();
        }
    }


    /**
     * Gets a python session object to use for interacting with python
     *
     * @return a PythonSession object
     * @throws WekaException if a problem occurs
     */
    protected RapidsSession getSession() throws WekaException {
        RapidsSession session = null;
        String pyCommand = m_pyCommand != null && m_pyCommand.length() > 0
                && !m_pyCommand.equalsIgnoreCase("default") ? m_pyCommand : null;
        String pyPath = m_pyPath != null && m_pyPath.length() > 0
                && !m_pyPath.equalsIgnoreCase("default") ? m_pyPath : null;
        String serverID = m_serverID != null && m_serverID.length() > 0
                && !m_serverID.equalsIgnoreCase("none") ? m_serverID : null;
        if (pyCommand != null) {
            if (!RapidsSession.pythonAvailable(pyCommand, serverID)) {
                // try to create this environment/server
                // System.err.println("Starting server: " + pyCommand + " " + serverID);
                if (!RapidsSession.initSession(pyCommand, serverID, pyPath, getDebug())) {
                    String envEvalResults =
                            RapidsSession.getPythonEnvCheckResults(pyCommand, serverID);
                    throw new WekaException("Was unable to start python environment ("
                            + pyCommand + ")\n\n" + envEvalResults);
                }
            }
            session = RapidsSession.acquireSession(pyCommand, serverID, this);
        } else {
            if (!RapidsSession.pythonAvailable()) {
                // try initializing
                if (!RapidsSession.initSession("python", getDebug())) {
                    String envEvalResults = RapidsSession.getPythonEnvCheckResults();
                    throw new WekaException(
                            "Was unable to start python environment:\n\n" + envEvalResults);
                }
            }
            session = RapidsSession.acquireSession(this);
        }

        return session;
    }

    /**
     * Release the python session
     *
     * @throws WekaException if a problem occurs
     */
    protected void releaseSession() throws WekaException {
        RapidsSession session = null;
        String pyCommand = m_pyCommand != null && m_pyCommand.length() > 0
                && !m_pyCommand.equalsIgnoreCase("default") ? m_pyCommand : null;
        String serverID = m_serverID != null && m_serverID.length() > 0
                && !m_serverID.equalsIgnoreCase("none") ? m_serverID : null;
        if (pyCommand != null) {
            if (RapidsSession.pythonAvailable(pyCommand, serverID)) {
                RapidsSession.releaseSession(pyCommand, serverID, this);
            }
        } else {
            if (RapidsSession.pythonAvailable()) {
                RapidsSession.releaseSession(this);
            }
        }
    }

    /**
     * Get the results of executing the environment check in python
     *
     * @return the results of executing the environment check
     * @throws WekaException if a problem occurs
     */
    protected String getPythonEnvCheckResults() throws WekaException {
        String result = "";
        String pyCommand = m_pyCommand != null && m_pyCommand.length() > 0
                && !m_pyCommand.equalsIgnoreCase("default") ? m_pyCommand : null;
        String serverID = m_serverID != null && m_serverID.length() > 0
                && !m_serverID.equalsIgnoreCase("none") ? m_serverID : null;

        if (pyCommand != null) {
            result = RapidsSession.getPythonEnvCheckResults(pyCommand, serverID);
        } else {
            result = RapidsSession.getPythonEnvCheckResults();
        }

        return result;
    }

    /**
     * Check to see if python is available for the user-specified environment.
     *
     * @return True if available, False otherwise
     */
    protected boolean pythonAvailable() {
        String pyCommand = m_pyCommand != null && m_pyCommand.length() > 0
                && !m_pyCommand.equalsIgnoreCase("default") ? m_pyCommand : null;
        String serverID = m_serverID != null && m_serverID.length() > 0
                && !m_serverID.equalsIgnoreCase("none") ? m_serverID : null;

        if (pyCommand != null) {
            return RapidsSession.pythonAvailable(pyCommand, serverID);
        }
        return RapidsSession.pythonAvailable();
    }

    /**
     * Build the classifier
     *
     * @param data set of instances serving as training data
     * @throws Exception if a problem occurs
     */
    @Override
    public void buildClassifier(Instances data) throws Exception {
        getCapabilities(true).testWithFail(data);
        m_zeroR = null;

        RapidsSession session = getSession();
        InstanceSender sender = new InstanceSender(session, m_sendingMethod, getDebug());

        if (m_modelHash == null) {
            m_modelHash = "" + hashCode();
        }

        data = new Instances(data);
        data.deleteWithMissingClass();
        m_zeroR = new ZeroR();
        m_zeroR.buildClassifier(data);
        m_classPriors = data.numInstances() > 0
                ? m_zeroR.distributionForInstance(data.instance(0))
                : new double[data.classAttribute().numValues()];
        if (data.numInstances() == 0 || data.numAttributes() == 1) {
            if (data.numInstances() == 0) {
                System.err
                        .println("No instances with non-missing class - using ZeroR model");
            } else {
                System.err.println("Only the class attribute is present in "
                        + "the data - using ZeroR model");
            }
            return;
        } else {
            m_zeroR = null;
        }

        if (data.classAttribute().isNominal()) {
            // check for empty classes
            AttributeStats stats = data.attributeStats(data.classIndex());
            m_nominalEmptyClassIndexes =
                    new boolean[data.classAttribute().numValues()];
            for (int i = 0; i < stats.nominalWeights.length; i++) {
                if (stats.nominalWeights[i] == 0) {
                    m_nominalEmptyClassIndexes[i] = true;
                }
            }
        }

        m_replaceMissing.setInputFormat(data);
        data = Filter.useFilter(data, m_replaceMissing);

        if (getUseSupervisedNominalToBinary()) {
            m_nominalToBinary =
                    new weka.filters.supervised.attribute.NominalToBinary();
        } else {
            m_nominalToBinary =
                    new weka.filters.unsupervised.attribute.NominalToBinary();
        }
        m_nominalToBinary.setInputFormat(data);
        data = Filter.useFilter(data, m_nominalToBinary);

        try {
            String learnerModule = m_learner.getModule();
            String learnerMethod = m_learner.toString();

            // transfer the data over to python
            sender.sendInstances(data, TRAINING_DATA_ID, getDebug());

            StringBuilder learnScript = new StringBuilder();

            learnScript.append("from cuml import ").append(learnerModule).append("\n")
                    .append("import cupy as cp\n")
                    .append("import cudf\n");

            learnScript
                    .append(MODEL_ID + m_modelHash + " = " + learnerModule + "."
                            + learnerMethod + "("
                            + (getLearnerOpts().length() > 0 ? getLearnerOpts() : "") + ")")
                    .append("\n");

//            learnScript.append("X, Y = cudf.from_pandas(X), cudf.from_pandas(Y)\n");
            if (m_learner.isClassifier()) {
                learnScript.append(MODEL_ID + m_modelHash + ".fit(X.astype('float32'), Y.astype('int32'))\n");
            } else if (m_learner.isRegressor()) {
                learnScript.append(MODEL_ID + m_modelHash + ".fit(X.astype('float32'), Y.astype('float32'))\n");
            } else {
                throw new WekaException("The learner " + m_learner.name() + "is not classifier nor regressor!");
            }

            List<String> outAndErr =
                    session.executeScript(learnScript.toString(), getDebug());
            if (outAndErr.size() == 2 && outAndErr.get(1).length() > 0) {
                if (m_continueOnSysErr) {
                    System.err.println(outAndErr.get(1));
                } else {
                    throw new Exception(outAndErr.get(1));
                }
            }

            m_learnerToString = session.getVariableValueFromPythonAsPlainString(
                    MODEL_ID + m_modelHash, getDebug());

            if (m_learner.removeModelFromPythonPostTrainPredict()) {
                String cleanUp = "del " + MODEL_ID + m_modelHash + "\n";
                outAndErr = session.executeScript(cleanUp, getDebug());

                if (outAndErr.size() == 2 && outAndErr.get(1).length() > 0) {
                    if (m_continueOnSysErr) {
                        System.err.println(outAndErr.get(1));
                    } else {
                        throw new Exception(outAndErr.get(1));
                    }
                }
            }
            // release session
        } finally {
            sender.release();
            releaseSession();
        }
    }

    private double[][] batchScoreWithZeroR(Instances insts) throws Exception {
        double[][] result = new double[insts.numInstances()][];

        for (int i = 0; i < insts.numInstances(); i++) {
            Instance current = insts.instance(i);
            result[i] = m_zeroR.distributionForInstance(current);
        }

        return result;
    }

    /**
     * Return the predicted probabilities for the supplied instance
     *
     * @param instance the instance to be classified
     * @return the predicted probabilities
     * @throws Exception if a problem occurs
     */
    @Override
    public double[] distributionForInstance(Instance instance) throws Exception {

        Instances temp = new Instances(instance.dataset(), 0);
        temp.add(instance);

        return distributionsForInstances(temp)[0];
    }

    /**
     * Return the predicted probabilities for the supplied instances
     *
     * @param insts the instances to get predictions for
     * @return the predicted probabilities for the supplied instances
     * @throws Exception if a problem occurs
     */
    @Override
    @SuppressWarnings("unchecked")
    public double[][] distributionsForInstances(Instances insts)
            throws Exception {
        if (m_zeroR != null) {
            return batchScoreWithZeroR(insts);
        }

        insts = Filter.useFilter(insts, m_replaceMissing);
        insts = Filter.useFilter(insts, m_nominalToBinary);
        Attribute classAtt = insts.classAttribute();
        // remove the class attribute
        Remove r = new Remove();
        r.setAttributeIndices("" + (insts.classIndex() + 1));
        r.setInputFormat(insts);
        insts = Filter.useFilter(insts, r);
        insts.setClassIndex(-1);

        double[][] results = null;
        RapidsSession session = null;
        InstanceSender sender = null;
        try {
            session = getSession();
            sender = new InstanceSender(session, m_sendingMethod, getDebug());
            sender.sendInstances(insts, TEST_DATA_ID, getDebug());
            StringBuilder predictScript = new StringBuilder();

            // check if model exists in python. If not, then transfer it over
            if (!session.checkIfPythonVariableIsSet(MODEL_ID + m_modelHash,
                    getDebug())) {
                throw new Exception("There is no model exist in Python environment!");
            }

            String learnerModule = m_learner.getModule();
            String learnerMethod = m_learner.toString();

//            predictScript.append("X = cudf.from_pandas(X)\n");
            predictScript
                    .append("from cuml." + learnerModule + " import " + learnerMethod).append("\n");

            predictScript.append("preds = " + MODEL_ID + m_modelHash + ".predict"
                    + (m_learner.producesProbabilities(m_learnerOpts) ? "_proba" : "")
                    + "(X)").append("\npreds = preds.values.tolist()\n");
            List<String> outAndErr =
                    session.executeScript(predictScript.toString(), getDebug());
            if (outAndErr.size() == 2 && outAndErr.get(1).length() > 0) {
                if (m_continueOnSysErr) {
                    System.err.println(outAndErr.get(1));
                } else {
                    throw new Exception(outAndErr.get(1));
                }
            }

            List<Object> preds = (List<Object>) session
                    .getVariableValueFromPythonAsJson("preds", getDebug());
            if (preds == null) {
                throw new Exception("Was unable to retrieve predictions from python");
            }

            if (preds.size() != insts.numInstances()) {
                throw new Exception(
                        "Learner did not return as many predictions as there "
                                + "are test instances");
            }

            results = new double[insts.numInstances()][];
            if (m_learner.producesProbabilities(m_learnerOpts)
                    && classAtt.isNominal()) {
                int j = 0;
                for (Object o : preds) {
                    List<Number> dist = (List<Number>) o;
                    double[] newDist = new double[classAtt.numValues()];
                    int k = 0;
                    for (int i = 0; i < newDist.length; i++) {
                        if (m_nominalEmptyClassIndexes[i]) {
                            continue;
                        }
                        newDist[i] = dist.get(k++).doubleValue();
                    }
                    try {
                        Utils.normalize(newDist);
                    } catch (IllegalArgumentException e) {
                        newDist = m_classPriors;
                        System.err.println(
                                "WARNING: " + e.getMessage() + ". Predicting using class priors");
                    }
                    results[j++] = newDist;
                }
            } else {
                if (classAtt.isNominal()) {
                    int j = 0;
                    for (Object o : preds) {
                        double[] dist = new double[classAtt.numValues()];
                        Number p = null;
                        if (o instanceof List) {
                            p = (Number) ((List) o).get(0);
                        } else {
                            p = (Number) o;
                        }
                        dist[p.intValue()] = 1.0;
                        results[j++] = dist;
                    }
                } else {
                    int j = 0;
                    for (Object o : preds) {
                        double[] dist = new double[1];
                        Number p = null;
                        if (o instanceof List) {
                            p = (Number) ((List) o).get(0);
                        } else {
                            p = (Number) o;
                        }
                        dist[0] = p.doubleValue();
                        results[j++] = dist;
                    }
                }
            }

            if (m_learner.removeModelFromPythonPostTrainPredict()) {
                String cleanUp = "del " + MODEL_ID + m_modelHash + "\n";
                outAndErr = session.executeScript(cleanUp, getDebug());

                if (outAndErr.size() == 2 && outAndErr.get(1).length() > 0) {
                    if (m_continueOnSysErr) {
                        System.err.println(outAndErr.get(1));
                    } else {
                        throw new Exception(outAndErr.get(1));
                    }
                }
            }

        } finally {
            sender.release();
            releaseSession();
        }

        return results;
    }

    /**
     * Get a textual description of this scheme
     *
     * @return a textual description of this scheme
     */
    public String toString() {
        if (m_zeroR != null) {
            return m_zeroR.toString();
        }
        if (m_learnerToString == null || m_learnerToString.length() == 0) {
            return "CuMLClassifier: model not built yet!";
        }
        return m_learnerToString;
    }

    /**
     * Main method for testing this class
     *
     * @param argv command line args
     */
    public static void main(String[] argv) {
        runClassifier(new CuMLClassifier(), argv);
    }
}
