package mulan.classifier.ensemble;

import mulan.classifier.MultiLabelLearner;
import mulan.classifier.MultiLabelOutput;

public class MajorityVoteEnsemble extends MultiLabelEnsembleLearner {

  public MajorityVoteEnsemble(MultiLabelLearner[] theClassifiers) {
    super(theClassifiers);
  }


  public MultiLabelOutput makePredictionInternal
                              (MultiLabelOutput[] classifierOutputs) {

    boolean[][] bipartitionVectors = new boolean[classifiers.length][];
    for (int i = 0; i < classifierOutputs.length; i++)
      bipartitionVectors[i] = classifierOutputs[i].getBipartition();

    int[] labelTallies = new int[bipartitionVectors[0].length];
    for (boolean[] bipartitionVector: bipartitionVectors)
      for (int label = 0; label < bipartitionVector.length; label++)
        labelTallies[label] += bipartitionVector[label] ? +1 : -1;

    boolean[] majorityVector = new boolean[labelTallies.length];
    for (int label = 0; label < labelTallies.length; label++)
      majorityVector[label] = (labelTallies[label] >= 0);

    return new MultiLabelOutput(majorityVector);
  }
}
