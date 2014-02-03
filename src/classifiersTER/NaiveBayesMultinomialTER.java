package classifiersTER;

import java.io.PrintStream;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;

import weka.classifiers.bayes.NaiveBayesMultinomial;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

/**
<!-- globalinfo-start -->
 * Class for building and using a multinomial Naive Bayes classifier with the application of different weightings developped at LIRMM <br/>
 * <br/>
 * **Reference vers l'article**<br/>
 * <br/>
 * The equations for this weightings:<br/>
 * <br/>
 *\\TODO<br/>
 * <br/>
 * 
 * <p/>
<!-- globalinfo-end -->
 *
<!-- technical-bibtex-start -->
 * BibTeX:
 * <pre>
 * \\TODO
 * }
 * </pre>
 * <p/>
<!-- technical-bibtex-end -->
 *
<!-- options-start -->
 * Valid options are: <p/>
 * 
 * <pre> -D
 *  If set, classifier is run in debug mode and
 *  may output additional info to the console</pre>
 * 
<!-- options-end -->
 *
 * @author Hadrien Negros (Hadrien.Negros@etud.univ-montp2.fr)
 * @author Mehdi Alijate (Mehdi.Alijate@etud.univ-montp2.fr)
 * @version $Revision: 1 $ 
 */
public class NaiveBayesMultinomialTER extends NaiveBayesMultinomial {

	/** for serialization */
	private static final long serialVersionUID = 1986672163986255572L;


	public void buildClassifier(Instances instances) throws Exception 
	{
		// can classifier handle the data?
		getCapabilities().testWithFail(instances);

		// remove instances with missing class
		instances = new Instances(instances);
		instances.deleteWithMissingClass();

		m_headerInfo = new Instances(instances, 0);
		m_numClasses = instances.numClasses();
		m_numAttributes = instances.numAttributes();
		m_probOfWordGivenClass = new double[m_numClasses][]; //Wij

		/*
		 * These variables represents the different values that we will need to
		 * calculate the weighting
		 */

		double[][] nbOfWordGivenClass = new double[m_numClasses][]; 
		for(int c = 0; c<m_numClasses; c++)
		{
			nbOfWordGivenClass[c] = new double[m_numAttributes];
			for(int att = 0; att<m_numAttributes; att++)
			{
				nbOfWordGivenClass[c][att] = 0;
			}
		}

		double[][] nbOfDocsContainingWordGivenClass = new double[m_numClasses][]; 
		for(int c = 0; c<m_numClasses; c++)
		{
			nbOfDocsContainingWordGivenClass[c] = new double[m_numAttributes];
			for(int att = 0; att<m_numAttributes; att++)
			{
				nbOfDocsContainingWordGivenClass[c][att] = 0;
			}
		}

		double[] nbOfClassesContainingWord = new double[m_numAttributes]; 
		for(int c = 0; c<m_numAttributes; c++)
		{
			nbOfClassesContainingWord[c] = 0;
		}
		HashMap<Integer,HashSet<Integer>> classesGivenWord = new HashMap<Integer,HashSet<Integer>>();
		for(int att = 0; att<m_numAttributes; att++)
		{
			classesGivenWord.put(att, new HashSet<Integer>());
		}		


		//Je ne sais pas du tout comment on va faire pour calculer ces 2 prochaines valeurs.
		double[] nbOfDocsNotInThisClass = new double[m_numClasses]; 
		for(int c = 0; c<m_numClasses; c++)
		{
			nbOfDocsNotInThisClass[c] = 0;
		}

		double[][] nbOfDocsNotInThisClassContainingWord = new double[m_numClasses][]; 
		for(int c = 0; c<m_numClasses; c++)
		{
			nbOfDocsNotInThisClassContainingWord[c] = new double[m_numAttributes];
			for(int att = 0; att<m_numAttributes; att++)
			{
				nbOfDocsContainingWordGivenClass[c][att] = 0;
			}		
		}


		//enumerate through the instances 
		Instance instance;
		int classIndex;
		double numOccurences;
		double[] docsPerClass = new double[m_numClasses];
		double[] wordsPerClass = new double[m_numClasses];

		java.util.Enumeration enumInsts = instances.enumerateInstances();
		while (enumInsts.hasMoreElements()) 
		{
			instance = (Instance) enumInsts.nextElement();
			classIndex = (int)instance.value(instance.classIndex());
			docsPerClass[classIndex] += instance.weight();

			for(int a = 0; a<instance.numValues(); a++)
				if(instance.index(a) != instance.classIndex())
				{
					if(!instance.isMissing(a))
					{

						numOccurences = instance.valueSparse(a) * instance.weight();
						if(numOccurences < 0)
							throw new Exception("Numeric attribute values must all be greater or equal to zero.");
						wordsPerClass[classIndex] += numOccurences;
						nbOfWordGivenClass[classIndex][instance.index(a)] += numOccurences;
						if(numOccurences > 0){
							nbOfDocsContainingWordGivenClass[classIndex][instance.index(a)] += 1;
							classesGivenWord.get(instance.index(a)).add(classIndex);
						}
					}
				} 
		}

		displayWeightings(System.out,instances,wordsPerClass,nbOfWordGivenClass,nbOfDocsContainingWordGivenClass,classesGivenWord);



	}
	private void displayWeightings(PrintStream out, Instances instances, double[] wordsPerClass,
			double[][] nbOfWordGivenClass,
			double[][] nbOfDocsContainingWordGivenClass,
			HashMap<Integer, HashSet<Integer>> classesGivenWord) {
		System.out.println("Words Per Class:\n");
		for(int c = 0; c<m_numClasses; c++)
		{

			out.println(instances.attribute(instances.classIndex()).value(c) +" : " +wordsPerClass[c]);
		}

		out.println("\nNb Words given Class:\n");
		for(int c = 0; c<m_numClasses; c++)
		{
			out.println(instances.attribute(instances.classIndex()).value(c)); 
			for(int att = 0; att<m_numAttributes; att++)
			{
				out.println("\t"+instances.attribute(att).name() +" : " +nbOfWordGivenClass[c][att]);

			}
		}
		out.println("\nNb Docs containing Word given Class:\n");
		for(int c = 0; c<m_numClasses; c++)
		{
			out.println(instances.attribute(instances.classIndex()).value(c)); 
			for(int att = 0; att<m_numAttributes; att++)
			{
				out.println("\t"+instances.attribute(att).name() +" : " +nbOfDocsContainingWordGivenClass[c][att]);

			}
		}
		out.println("\nClasses Given Word:\n");
		for(int att = 0; att<m_numAttributes; att++)
		{
			out.print(instances.attribute(att).name() + " : ");
			for(int c : classesGivenWord.get(att)){
				out.print(instances.attribute(instances.classIndex()).value(c) +  " ");
			}
			out.println();
		}

	}
	/**
	 * Main method for testing this class.
	 *
	 * @param argv the options
	 * @throws Exception 
	 */
	public static void main(String [] argv) {
		NaiveBayesMultinomialTER test = new NaiveBayesMultinomialTER();
		try {
			DataSource ds = new DataSource(argv[1]);
			Instances i = ds.getDataSet();
			i.setClassIndex(i.numAttributes()-1);
			test.buildClassifier(i);
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

}
