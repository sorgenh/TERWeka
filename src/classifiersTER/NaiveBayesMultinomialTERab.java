package weka.classifiers.bayes;


import java.io.PrintStream;
import java.util.ArrayList;
import java.util.Enumeration;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Vector;

import weka.classifiers.bayes.NaiveBayesMultinomial;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Option;
import weka.core.OptionHandler;
import weka.core.Utils;

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
 * <pre> -A &lt;alphaString&gt;
 *  Set the alpha value. (default = 1).</pre>
 * 
 * <pre> -B &lt;betaString&gt;
 *  Set the beta value. (default = 0).</pre>
 *  
 * <pre> -D
 *  If set, classifier is run in debug mode and
 *  may output additional info to the console</pre>
 * 
<!-- options-end -->
 *
 * @author Hadrien Negros (Hadrien.Negros@etud.univ-montp2.fr)
 * @author Mehdi Alijate (Mehdi.Alijate@etud.univ-montp2.fr)
 * @author Batoul Turki (Batoul.Turki@etud.univ-montp2.fr)
 * @version $Revision: 8 $ 
 */
public class NaiveBayesMultinomialTERab extends NaiveBayesMultinomial implements OptionHandler{

	/** for serialization */
	private static final long serialVersionUID = 1986672163986255572L;

	protected double m_alpha=1;
	protected double m_beta = 0;

	@Override
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

		@SuppressWarnings("rawtypes")
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


		double temp=0;
		double[][] intra1 = new double[m_numClasses][];
		double[][] intra2 = new double[m_numClasses][];
		double[][] inter1 = new double[m_numClasses][];
		double[][] inter2 = new double[m_numClasses][];
		double maxIntra1 = 0;
		double maxIntra2 = 0;
		double maxInter1 = 0;
		double maxInter2 = 0;
		int nbClassContaningWord;

		//Calcul des valeurs intra1, intra2, inter1, inter2
		for(int j = 0; j<m_numClasses; j++)
		{
			intra1[j] = new double[m_numAttributes];
			intra2[j] = new double[m_numAttributes];
			inter1[j] = new double[m_numAttributes];
			inter2[j] = new double[m_numAttributes];
			for(int i = 0; i<m_numAttributes; i++)
			{
				if(classesGivenWord.get(i).contains(j))
					nbClassContaningWord = classesGivenWord.get(i).size()-1;
				else
					nbClassContaningWord = classesGivenWord.get(i).size();
				intra1[j][i] = nbOfWordGivenClass[j][i] / wordsPerClass[j];
				intra2[j][i] = (nbOfDocsContainingWordGivenClass[j][i] / nbDocsPerClass[j]);
				inter1[j][i] = (Math.log(((double)m_numClasses)/((double)(nbClassContaningWord+1)))/Math.log(2));
				for(int x = 0; x<m_numClasses; x++)
				{
					if(x != j)
						temp += nbOfDocsContainingWordGivenClass[x][i];	
				}
				inter2[j][i] = (Math.log((numDocs - docsGivenClass.get(j).size()+1)/(temp+1)))/Math.log(2);

				if(intra1[j][i]> maxIntra1)
					maxIntra1 = intra1[j][i];

				if(intra2[j][i]> maxIntra2)
					maxIntra2 = intra2[j][i];

				if(inter1[j][i]> maxInter1)
					maxInter1 = inter1[j][i];

				if(inter2[j][i]> maxInter2)
					maxInter2 = inter2[j][i];

			}
		}	
		for(int j = 0; j<m_numClasses; j++)
		{
			m_probOfWordGivenClass[j] = new double[m_numAttributes];
			for(int i = 0; i<m_numAttributes; i++)
			{
				m_probOfWordGivenClass[j][i] = 
						(m_alpha * (intra1[j][i]/maxIntra1) 
								+ (1-m_alpha)* (intra2[j][i]/maxIntra2))
								* (m_beta*(inter1[j][i]/maxInter1) 
										+ (1-m_beta) *(inter2[j][i]/maxInter2));
			}
		}


		/*
	      calculating Pr(H)
	      NOTE: Laplace estimator introduced in case a class does not get mentioned in the set of 
	      training instances
		 */
		m_probOfClass = new double[m_numClasses];
		for(int h=0; h<m_numClasses; h++)
			m_probOfClass[h] = (nbDocsPerClass[h] + 1)/numDocs; 

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


		out.println("\nPondŽration a="+m_alpha+" b="+m_beta+" :Prob of Words given Class (Wij):\n");
		for(int c = 0; c<m_numClasses; c++)
		{
			out.println(instances.attribute(instances.classIndex()).value(c)); 
			for(int att = 0; att<m_numAttributes; att++)
			{
				out.println("\t"+instances.attribute(att).name() +" : " +probofWordGivenClass[c][att]);

			}
		}


	}
	@Override
	

	public Enumeration<Option> listOptions() {
		Vector<Option> newVector = new Vector<Option>();

		newVector.add(new Option("\tSet value of alpha "
				+"(default = 1).",
				"A", 1, "-A <alphaString>"));
		newVector.add(new Option("\tSet value of beta "
				+"(default = 0).",
				"B", 1, "-B <betaString>"));
		return newVector.elements();
	}

	/**
	 *
	 * @param options the list of options as an array of strings
	 * @throws Exception if an option is not supported
	 */
	public void setOptions(String[] options) throws Exception {
		setDebug(Utils.getFlag('D', options));

		String alphaString = Utils.getOption('A', options);
		if (alphaString.length() != 0) {
			setAlpha(Double.parseDouble(alphaString));
		} 


		String betaString = Utils.getOption('B', options);
		if (betaString.length() != 0) {
			setBeta(Double.parseDouble(betaString));
		}

	}

	 /**
	   * Gets the current settings of the classifier.
	   *
	   * @return an array of strings suitable for passing to setOptions
	   */
	  public String [] getOptions() {
		
	    ArrayList<String> options = new ArrayList<String>();
	   
	    
	      options.add("-A");
	      options.add(""+getAlpha());

	      options.add("-B");
	      options.add(""+getBeta());


	    return options.toArray(new String[1]);
	  }
	  
	  public double getAlpha(){
		  return m_alpha;
	  }
	 public void setAlpha(double a){
		  m_alpha = a;
	  }
	  public String alphaTipText(){
		  return "Sets the value of alpha.";
	  }
	  
	 public double getBeta(){
		  return m_beta;
	  }
	 public void setBeta(double b){
		  m_beta = b;
	  }
	 public String betaTipText(){
		  return "Sets the value of beta.";
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
		runClassifier(new NaiveBayesMultinomialTERab(), argv);
	}

}
