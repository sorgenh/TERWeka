package weka.classifiers.bayes;


import java.util.Enumeration;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Vector;

import weka.classifiers.bayes.NaiveBayesMultinomial;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Option;
import weka.core.SelectedTag;
import weka.core.Tag;
import weka.core.TechnicalInformation;
import weka.core.Utils;
import weka.core.TechnicalInformation.Field;
import weka.core.TechnicalInformation.Type;


/**
<!-- globalinfo-start -->
 * Class for building and using a multinomial Naive Bayes classifier with the application of four different weightings developed at LIRMM <br/>
 * <br/>
 * <br/>
 * The equations for this weightings:<br/>
 * <br/>
 *The weightings for this classifier can be selected from the properties, and are calculated as follows :<br/>
 *Weighting 1 : W_Tf-Class[ij] = intra-classe(Tf)[ij] × inter-classe(class)[ij] <br/>
 *Weighting 2 : W_Df-Class[ij] = intra-classe(Df)[ij] × inter-classe(class)[ij] <br/>
 *Weighting 3 : W_Tf-Doc[ij] = intra-classe(Tf)[ij] x inter-classe(Doc)[ij]<br/>
 *Weighting 4 : W_Df-Doc[ij] = intra-classe(Df)[ij] x inter-classe(Doc)[ij] <br/>
 *where intra-classe(Tf)[ij] = [(number of occurrences of the word i in the class j)/(the total number of words in class j)]+1 <br/>
 *where intra-classe(Df)[ij] = [(number of documents containing the word i in the class j)/(the total number of documents in class j)]+1 <br/>
 *where inter-classe(class)[ij] = Log2[(number of classes excluding class j)+1 / (the total number of classes exluding class j containing the word i)+1] <br/>
 *where inter-classe(Doc)[ij] = Log2[(number of documents out of class j)+1 / (the total number of documents out of class j containing the word i)+1] <br/>
 *where j is a class and i is a word <br/>
 * <br/>
 * 
 * <p/>
<!-- globalinfo-end -->

<!-- options-start -->
 * Valid options are: <p/>
 * 
 * <pre> -D
 *  If set, classifier is run in debug mode and
 *  may output additional info to the console</pre>
 *  * <pre> -W &lt;Weighting&gt;
 *  Choose the Weighting (default: 1)
 *    1 = W_Tf-Class
 *    2 = W_Df-Class
 *    3 = W_Tf-Doc
 *    4 = W_Df-Doc</pre>
<!-- options-end -->
 *
 * @author Hadrien Negros (Hadrien.Negros@etud.univ-montp2.fr)
 * @author Mehdi Alijate (Mehdi.Alijate@etud.univ-montp2.fr)
 * @author Batoul Turki (Turki.Batoul@etud.univ-montp2.fr)
 * @version $Revision: 7 $ 
 */
public class NaiveBayesMultinomialTER extends NaiveBayesMultinomial {

	/** for serialization */
	private static final long serialVersionUID = 1986672163986255572L;
	  protected int m_Weighting=1;
	  /** model types */
	  public static final int W1 = 1;
	  public static final int W2 = 2;
	  public static final int W3 = 3;
	  public static final int W4 = 4;


	  /** possible model types. */
	  public static final Tag [] TAGS_MODEL = {
	    new Tag(W1, "W_Tf-Class"),
	    new Tag(W2, "W_Df-Class"),
	    new Tag(W3, "W_Tf-Doc"),
	    new Tag(W4, "W_Df-Doc")
	  };
/**
	   * Returns a string describing this classifier
	   * @return a description of the classifier suitable for
	   * displaying in the explorer/experimenter gui
	   */
	  public String globalInfo() {
	    return 
	        "Class for building and using a multinomial Naive Bayes classifier with the application of four different weightings developped at LIRMM. "
	      + "For more information see,\n\n"
	      + getTechnicalInformation().toString() +"\n"
	      + "The weightings for this classifier can be selected from the properties, and are calculated as follows :\n\n"
	      + "Weighting 1 : W_Tf-Class[ij] = intra-classe(Tf)[ij] × inter-classe(class)[ij] \n\n"
	      + "Weighting 2 : W_Df-Class[ij] = intra-classe(Df)[ij] × inter-classe(class)[ij] \n\n"
	      + "Weighting 3 : W_Tf-Doc[ij] = intra-classe(Tf)[ij] x inter-classe(Doc)[ij]\n\n"
	      + "Weighting 4 : W_Df-Doc[ij] = intra-classe(Df)[ij] x inter-classe(Doc)[ij] \n\n"
	      + "where intra-classe(Tf)[ij] = [(number of occurrences of the word i in the class j)+1/(the total number of words in class j)+1] \n\n"
	      + "where intra-classe(Df)[ij] = [(number of documents containing the word i in the class j)+1/(the total number of documents in class j)+1] \n\n"
	      + "where inter-classe(class)[ij] = Log2[(number of classes excluding class j)+1 / (the total number of classes exluding class j containing the word i)+1] \n\n"
	      + "where inter-classe(Doc)[ij] = Log2[(number of documents out of class j)+1 / (the total number of documents out of class j containing the word i)+1]\n\n"
	      + "where j is a class and i is a word.";
	  } 
	 

	  public TechnicalInformation getTechnicalInformation() {
		    TechnicalInformation 	result;
		    
		    result = new TechnicalInformation(Type.INPROCEEDINGS);
		    result.setValue(Field.AUTHOR, "ALIJATE Mehdi, NEGROS Hadrien, TURKI Batoul");
		    result.setValue(Field.YEAR, "TER M2, Montpellier 2 University France - 2014");
		    result.setValue(Field.TITLE, "Paper : 'New weightings adapted to the classification of small volumes of textual data'");
		    result.setValue(Field.BOOKTITLE, "Supervised by Flavien Bouillot, Pascal Poncelet and Mathieu Roche ");
		    
		    return result;
		  }

	//private final int weightingType = 1;

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
		//System.out.println("Nb docs: "+numDocs);

		
		double inter,intra,temp=0;
		int nbClassContaningWord;

		if(m_Weighting==1){
			for(int j = 0; j<m_numClasses; j++)
		{
			m_probOfWordGivenClass[j] = new double[m_numAttributes];
			for(int i = 0; i<m_numAttributes; i++)
			{
				if(classesGivenWord.get(i).contains(j))
					nbClassContaningWord = classesGivenWord.get(i).size()-1;
				else
					nbClassContaningWord = classesGivenWord.get(i).size();
				intra = ((nbOfWordGivenClass[j][i]+1) / (wordsPerClass[j]+1));
				inter = (Math.log(((double)m_numClasses)/((double)(nbClassContaningWord+1)))/Math.log(2));
				m_probOfWordGivenClass[j][i] = intra*inter ;
			}
		}
		}
		
		if(m_Weighting==2){
			for(int j = 0; j<m_numClasses; j++)
			{
				m_probOfWordGivenClass[j] = new double[m_numAttributes];
				for(int i = 0; i<m_numAttributes; i++)
				{
					if(classesGivenWord.get(i).contains(j))
						nbClassContaningWord = classesGivenWord.get(i).size()-1;
					else
						nbClassContaningWord = classesGivenWord.get(i).size();
					intra = (((nbOfDocsContainingWordGivenClass[j][i])+1) / (nbDocsPerClass[j]+1)) ;
					inter = (Math.log(((double)m_numClasses)/((double)(nbClassContaningWord+1)))/Math.log(2));
					m_probOfWordGivenClass[j][i] = intra*inter ;
				}
			}

		}
		if(m_Weighting==3){
			for(int j = 0; j<m_numClasses; j++)
			{
				m_probOfWordGivenClass[j] = new double[m_numAttributes];
				for(int i = 0; i<m_numAttributes; i++)
				{
					intra = (((nbOfWordGivenClass[j][i])+1) /((wordsPerClass[j])+1));
					for(int x = 0; x<m_numClasses; x++)
					{
						if(x != j)
							temp += nbOfDocsContainingWordGivenClass[x][i];					}
					inter = (Math.log((numDocs - (double)docsGivenClass.get(j).size())/(temp+1)))/Math.log(2);
					m_probOfWordGivenClass[j][i] = intra*inter ;
					temp = 0;

				}
			}

		}
		if(m_Weighting==4){
			
			for(int j = 0; j<m_numClasses; j++)
			{
				m_probOfWordGivenClass[j] = new double[m_numAttributes];
				for(int i = 0; i<m_numAttributes; i++)
				{
					intra = ((nbOfDocsContainingWordGivenClass[j][i]+1) / (nbDocsPerClass[j]+1)) ;
					for(int x = 0; x<m_numClasses; x++)
					{
						if(x != j)
							temp += nbOfDocsContainingWordGivenClass[x][i];
					}
					inter = (Math.log((numDocs - (double)docsGivenClass.get(j).size())/(temp+1)))/Math.log(2);
					m_probOfWordGivenClass[j][i] = intra*inter ;
					temp = 0;
				}
			}
		}
		
		
		m_probOfClass = new double[m_numClasses];
		for(int h=0; h<m_numClasses; h++)
			m_probOfClass[h] = (double)(nbDocsPerClass[h] + 1)/numDocs; 

	}
	
	  public Enumeration<Option> listOptions() {
		    Vector<Option> newVector = new Vector<Option>(3);
		    

		    
		    newVector.addElement(new Option("\tChoise a weighting : "+
		                                    " 0 for W_Tf-class, 1 for W_Df-Class, 2 for W_Tf-Doc and 3 for W_Df-Doc",
		                                    "W",1,"-W <Weighting>"));
		    
		    return newVector.elements();
		  }
	
	  public void setOptions(String[] options) throws Exception {

		   String optionString = Utils.getOption('W', options);
		  
		    if (optionString.length() != 0) {
		      setWeighting(new SelectedTag(Integer.parseInt(optionString), TAGS_MODEL));
		
		    }
		          
		    
		    Utils.checkForRemainingOptions(options);
			
		  } 
	
	
	

	public String[] getOptions() {
		    String[] options = new String[11];
		    int current = 0;
		    
		    options[current++] = "-W"; 
		
		    options[current++] = ""+getWeighting().getSelectedTag().getID();
		    while (current < options.length) {
		        options[current++] = "";
		      } 
		      return options;
		    } 

	public SelectedTag getWeighting() {
	    return new SelectedTag(m_Weighting, TAGS_MODEL);
	  } 

	 
	  public void setWeighting(SelectedTag newMethod){
	    if (newMethod.getTags() == TAGS_MODEL) {
	      int c = newMethod.getSelectedTag().getID();
	      if (c==1 || c==2 || c==3 || c==4) {
	    	 m_Weighting = c;
	      } else  {
	        throw new IllegalArgumentException("No Weitghting selected, -W value should be: 1,2,3 or 4 " ); 
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

			runClassifier(new NaiveBayesMultinomialTER(), argv);
	}

}
