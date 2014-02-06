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

	private final int weightingType = 4;

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
		double numDocs = instances.sumOfWeights();
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
				nbOfWordGivenClass[c][att] = 1;
			}
		}

		double[][] nbOfDocsContainingWordGivenClass = new double[m_numClasses][]; 
		for(int c = 0; c<m_numClasses; c++)
		{
			nbOfDocsContainingWordGivenClass[c] = new double[m_numAttributes];
			for(int att = 0; att<m_numAttributes; att++)
			{
				nbOfDocsContainingWordGivenClass[c][att] = 1;
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


		HashMap<Integer,HashSet<Integer>> docsNotInThisClassContainingWord = new HashMap<Integer,HashSet<Integer>>(); 
		for(int att = 0; att<m_numAttributes; att++)
		{
			docsNotInThisClassContainingWord.put(att, new HashSet<Integer>());
		}
		//Représente tout les documents par classe
		HashMap<Integer,HashSet<Integer>> docsGivenClass = new HashMap<Integer,HashSet<Integer>>(); 
		for(int att = 0; att<m_numClasses; att++)
		{
			docsGivenClass.put(att, new HashSet<Integer>());
		}
		//Représente tout les documents contenant le mot
		HashMap<Integer,HashSet<Integer>> docsGivenWord = new HashMap<Integer,HashSet<Integer>>(); 
		for(int att = 0; att<m_numAttributes; att++)
		{
			docsGivenWord.put(att, new HashSet<Integer>());
		}





		//enumerate through the instances 
		Instance instance;
		int classIndex;
		double numOccurences;
		double[] nbDocsPerClass = new double[m_numClasses];
		double[] wordsPerClass = new double[m_numClasses];

		java.util.Enumeration enumInsts = instances.enumerateInstances();
		int ndoc=0;
		while (enumInsts.hasMoreElements()) 
		{
			instance = (Instance) enumInsts.nextElement();
			classIndex = (int)instance.value(instance.classIndex());
			nbDocsPerClass[classIndex] += instance.weight();
			docsGivenClass.get(classIndex).add(ndoc);

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
							nbOfDocsContainingWordGivenClass[classIndex][instance.index(a)] += instance.weight();
							classesGivenWord.get(instance.index(a)).add(classIndex);
							docsGivenWord.get(instance.index(a)).add(ndoc);
						}
					}
				} 

			ndoc++;
		}
		System.out.println("Nb docs: "+numDocs);

		
		double inter,intra,temp=0;
		int nbClassContaningWord;

		//On a toute les valeurs necessaires pour les weightings normalement
		switch(weightingType){
		case 1:
			//Premiere ponderation:
			for(int j = 0; j<m_numClasses; j++)
			{
				m_probOfWordGivenClass[j] = new double[m_numAttributes];
				for(int i = 0; i<m_numAttributes; i++)
				{
					if(classesGivenWord.get(i).contains(j))
						nbClassContaningWord = classesGivenWord.get(i).size()-1;
					else
						nbClassContaningWord = classesGivenWord.get(i).size();
					intra = nbOfWordGivenClass[j][i] / wordsPerClass[j];
					inter = (Math.log(((double)m_numClasses)/((double)(nbClassContaningWord+1)))/Math.log(2));
					m_probOfWordGivenClass[j][i] = intra*inter ;
				}
			}
			
			break;
		//Deuxieme Ponderation	
		case 2:
			for(int j = 0; j<m_numClasses; j++)
			{
				m_probOfWordGivenClass[j] = new double[m_numAttributes];
				for(int i = 0; i<m_numAttributes; i++)
				{
					if(classesGivenWord.get(i).contains(j))
						nbClassContaningWord = classesGivenWord.get(i).size()-1;
					else
						nbClassContaningWord = classesGivenWord.get(i).size();
					intra = (nbOfDocsContainingWordGivenClass[j][i] / nbDocsPerClass[j]) ;
					inter = (Math.log(((double)m_numClasses)/((double)(nbClassContaningWord+1)))/Math.log(2));
					m_probOfWordGivenClass[j][i] = intra*inter ;
				}
			}
			
			break;
		//Troisieme Ponderation
		case 3:
			for(int j = 0; j<m_numClasses; j++)
			{
				m_probOfWordGivenClass[j] = new double[m_numAttributes];
				for(int i = 0; i<m_numAttributes; i++)
				{
					intra = nbOfWordGivenClass[j][i] / wordsPerClass[j];
					for(int x = 0; x<m_numClasses; x++)
					{
						if(x != j)
							temp += nbOfDocsContainingWordGivenClass[x][i];					}
					inter = (Math.log((numDocs - (double)docsGivenClass.get(j).size())/temp))/Math.log(2);
					m_probOfWordGivenClass[j][i] = intra*inter ;
					temp = 0;

				}
			}

			break;
		case 4:
			for(int j = 0; j<m_numClasses; j++)
			{
				m_probOfWordGivenClass[j] = new double[m_numAttributes];
				for(int i = 0; i<m_numAttributes; i++)
				{
					intra = (nbOfDocsContainingWordGivenClass[j][i] / nbDocsPerClass[j]) ;
					for(int x = 0; x<m_numClasses; x++)
					{
						if(x != j)
							temp += nbOfDocsContainingWordGivenClass[x][i];
					}
					inter = (Math.log((numDocs - (double)docsGivenClass.get(j).size())/temp))/Math.log(2);
					m_probOfWordGivenClass[j][i] = intra*inter ;
					temp = 0;
				}
			}

			break;
		default:
			System.err.println("No weighting type specified.");

		}
	
		/*
	      calculating Pr(H)
	      NOTE: Laplace estimator introduced in case a class does not get mentioned in the set of 
	      training instances
		 */
		m_probOfClass = new double[m_numClasses];
		for(int h=0; h<m_numClasses; h++)
			m_probOfClass[h] = (double)(nbDocsPerClass[h] + 1)/numDocs; 

		displayWeightings(System.out,instances,wordsPerClass,nbOfWordGivenClass,nbOfDocsContainingWordGivenClass,classesGivenWord
				,docsGivenClass,docsGivenWord,m_probOfWordGivenClass);


	}
	private void displayWeightings(PrintStream out, Instances instances, double[] wordsPerClass,
			double[][] nbOfWordGivenClass,
			double[][] nbOfDocsContainingWordGivenClass,
			HashMap<Integer, HashSet<Integer>> classesGivenWord, HashMap<Integer, HashSet<Integer>> docsGivenClass, HashMap<Integer, HashSet<Integer>> docsGivenWord, double[][] probofWordGivenClass) {
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
		out.println("\nDocs Given Class:\n");
		for(int c = 0; c<m_numClasses; c++)
		{
			out.print(instances.attribute(instances.classIndex()).value(c) + " : ");
			for(int doc : docsGivenClass.get(c)){
				out.print(doc  +  " ");
			}
			out.println();
		}
		out.println("\nDocs Given Word:\n");
		for(int att = 0; att<m_numAttributes; att++)
		{
			out.print(instances.attribute(att).name() + " : ");
			for(int doc : docsGivenWord.get(att)){
				out.print(doc  +  " ");
			}
			out.println();
		}


		out.println("\nPondŽration "+weightingType+" :Prob of Words given Class (Wij):\n");
		for(int c = 0; c<m_numClasses; c++)
		{
			out.println(instances.attribute(instances.classIndex()).value(c)); 
			for(int att = 0; att<m_numAttributes; att++)
			{
				out.println("\t"+instances.attribute(att).name() +" : " +probofWordGivenClass[c][att]);

			}
		}


	}
	/**
	 * Main method for testing this class.
	 *
	 * @param argv the options
	 * @throws Exception 
	 */
	public static void main(String [] argv) {


//		NaiveBayesMultinomialTER test = new NaiveBayesMultinomialTER();
//		try {
//			DataSource ds = new DataSource(argv[1]);
//			Instances i = ds.getDataSet();
//			i.setClassIndex(i.numAttributes()-1);
//			test.buildClassifier(i);
//		} catch (Exception e) {
//			e.printStackTrace();
//		}
			runClassifier(new NaiveBayesMultinomialTER(), argv);
	}

}
