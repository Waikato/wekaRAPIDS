#!/bin/sh

set -x
mkdir -p log
# WEKA
(time weka -main weka.Run weka.classifiers.trees.RandomForest -P 100 -I 100 -num-slots 1 -K 0 -M 1.0 -V 0.001 -S 1 -depth 16 -t /home/justinliu/Projects/wekaRAPIDS/test/RDG1_1m.arff) 2>&1 |tee log/weka-RDG1_1m-RF.log
(time weka -main weka.Run weka.classifiers.trees.RandomForest -P 100 -I 100 -num-slots 1 -K 0 -M 1.0 -V 0.001 -S 1 -depth 16 -t /home/justinliu/Projects/wekaRAPIDS/test/RDG1_2m.arff) 2>&1 |tee log/weka-RDG1_2m-RF.log
(time weka -main weka.Run weka.classifiers.trees.RandomForest -P 100 -I 100 -num-slots 1 -K 0 -M 1.0 -V 0.001 -S 1 -depth 16 -t /home/justinliu/Projects/wekaRAPIDS/test/RDG1_5m.arff) 2>&1 |tee log/weka-RDG1_5m-RF.log
(time weka -main weka.Run weka.classifiers.trees.RandomForest -P 100 -I 100 -num-slots 1 -K 0 -M 1.0 -V 0.001 -S 1 -depth 16 -t /home/justinliu/Projects/wekaRAPIDS/test/RDG1_10m.arff) 2>&1 |tee log/weka-RDG1_10m-RF.log
(time weka -main weka.Run weka.classifiers.trees.RandomForest -P 100 -I 100 -num-slots 1 -K 0 -M 1.0 -V 0.001 -S 1 -depth 16 -t /home/justinliu/Projects/wekaRAPIDS/test/RDG1_5m_20a.arff) 2>&1 |tee log/weka-RDG1_5m_20a-RF.log
(time weka -main weka.Run weka.classifiers.trees.RandomForest -P 100 -I 100 -num-slots 1 -K 0 -M 1.0 -V 0.001 -S 1 -depth 16 -t /home/justinliu/Projects/wekaRAPIDS/test/RBFa5k.arff) 2>&1 |tee log/weka-RBFa5k-RF.log
(time weka -main weka.Run weka.classifiers.trees.RandomForest -P 100 -I 100 -num-slots 1 -K 0 -M 1.0 -V 0.001 -S 1 -depth 16 -t /home/justinliu/Projects/wekaRAPIDS/test/RBFa5kn1k.arff) 2>&1 |tee log/weka-RBFa5kn1k-RF.log
(time weka -main weka.Run weka.classifiers.trees.RandomForest -P 100 -I 100 -num-slots 1 -K 0 -M 1.0 -V 0.001 -S 1 -depth 16 -t /home/justinliu/Projects/wekaRAPIDS/test/RBFa5kn5k.arff) 2>&1 |tee log/weka-RBFa5kn5k-RF.log


# RAPIDS
(time weka -memory 48g -main weka.Run weka.classifiers.rapids.CuMLClassifier -learner RandomForestClassifier -t $(pwd)/test/RDG1_1m.arff -py-command python -output-debug-info) 2>&1 |tee log/rapid-RDG1_1m-RF.log
(time weka -memory 48g -main weka.Run weka.classifiers.rapids.CuMLClassifier -learner RandomForestClassifier -t $(pwd)/test/RDG1_2m.arff -py-command python -output-debug-info) 2>&1 |tee log/rapid-RDG1_2m-RF.log
(time weka -memory 48g -main weka.Run weka.classifiers.rapids.CuMLClassifier -learner RandomForestClassifier -t $(pwd)/test/RDG1_5m.arff -py-command python -output-debug-info) 2>&1 |tee log/rapid-RDG1_5m-RF.log
(time weka -memory 48g -main weka.Run weka.classifiers.rapids.CuMLClassifier -learner RandomForestClassifier -t $(pwd)/test/RDG1_10m.arff -py-command python -output-debug-info) 2>&1 |tee log/rapid-RDG1_10m-RF.log
(time weka -memory 48g -main weka.Run weka.classifiers.rapids.CuMLClassifier -learner RandomForestClassifier -t $(pwd)/test/RDG1_5m_20a.arff -py-command python -output-debug-info) 2>&1 |tee log/rapid-RDG1_5m_20a-RF.log
(time weka -memory 48g -main weka.Run weka.classifiers.rapids.CuMLClassifier -learner RandomForestClassifier -t $(pwd)/test/RBFa5k.arff -py-command python -output-debug-info) 2>&1 |tee log/rapid-RBFa5k-RF.log
(time weka -memory 48g -main weka.Run weka.classifiers.rapids.CuMLClassifier -learner RandomForestClassifier -t $(pwd)/test/RBFa5kn1k.arff -py-command python -output-debug-info) 2>&1 |tee log/rapid-RBFa5kn1k-RF.log
(time weka -memory 48g -main weka.Run weka.classifiers.rapids.CuMLClassifier -learner RandomForestClassifier -t $(pwd)/test/RBFa5kn5k.arff -py-command python -output-debug-info) 2>&1 |tee log/rapid-RBFa5kn5k-RF.log
