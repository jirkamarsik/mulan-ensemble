JFLAGS = -g
JC = javac

.SUFFIXES: .java .class

.java.class:
	$(JC) $(JFLAGS) $*.java

CLASSES = \
	mulan/classifier/ensemble/MultiLabelEnsembleLearner.java \
	mulan/classifier/ensemble/IntersectionEnsemble.java \
	mulan/classifier/ensemble/UnionEnsemble.java \
	mulan/classifier/ensemble/MajorityVoteEnsemble.java \
	mulan/classifier/ensemble/MinimumEnsemble.java \
	mulan/classifier/ensemble/MaximumEnsemble.java \
	mulan/classifier/ensemble/MeanEnsemble.java \
	mulan/classifier/ensemble/TopKEnsemble.java

default: classes

classes: $(CLASSES:.java=.class)

clean:
	$(RM) $(CLASSES:.java=.class)
