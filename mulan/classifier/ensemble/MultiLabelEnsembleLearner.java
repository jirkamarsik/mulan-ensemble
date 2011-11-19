package mulan.classifier.ensemble;

import mulan.classifier.MultiLabelLearner;
import mulan.classifier.MultiLabelOutput;
import mulan.data.MultiLabelInstances;
import weka.core.Instance;
import weka.core.SerializedObject;

public abstract class MultiLabelEnsembleLearner implements MultiLabelLearner {

  protected MultiLabelLearner[] classifiers;

  public MultiLabelEnsembleLearner(MultiLabelLearner[] theClassifiers) {
    classifiers = theClassifiers;
  }

  public abstract MultiLabelOutput makePredictionInternal
                                    (MultiLabelOutput[] classifierOutputs);

  public MultiLabelOutput makePrediction(Instance instance) throws Exception
  {
    MultiLabelOutput[] classifierOutputs = new MultiLabelOutput[classifiers.length];
    for (int i = 0; i < classifiers.length; i++)
      classifierOutputs[i] = classifiers[i].makePrediction(instance);
    return makePredictionInternal(classifierOutputs);
  }

  public void build(MultiLabelInstances instances) throws Exception {
    for (int i = 0; i < classifiers.length; i++)
      classifiers[i].build(instances);
  }

  public boolean isUpdatable() {
    return false;
  }

  public MultiLabelLearner makeCopy() throws Exception {
    return (MultiLabelLearner) new SerializedObject(this).getObject();
  }
}
