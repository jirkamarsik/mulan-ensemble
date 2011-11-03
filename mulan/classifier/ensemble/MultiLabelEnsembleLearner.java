package mulan.classifier.ensemble;

import mulan.classifier.MultiLabelLearner;
import mulan.classifier.MultiLabelOutput;
import mulan.data.MultiLabelInstances;
import weka.core.Instance;
import weka.core.SerializedObject;

public abstract class MultiLabelEnsembleLearner implements MultiLabelLearner {

  private MultiLabelLearner[] classifiers;

  public MultiLabelEnsembleLearner(MultiLabelLearner[] theClassifiers) {
    classifiers = theClassifiers;
  }

  public abstract MultiLabelOutput makePrediction(Instance instance);

  public void build(MultiLabelInstances instances) throws Exception {
    for (int i = 0; i < classifiers.length; i++)
      classifiers[i].build(instances);
  }

  public boolean isUpdatable() {
    boolean updatable = true;

    for (int i = 0; i < classifiers.length; i++)
      if (!classifiers[i].isUpdatable())
        updatable = false;

    return updatable;
  }

  public MultiLabelLearner makeCopy() throws Exception {
    return (MultiLabelLearner) new SerializedObject(this).getObject();
  }
}
