package classifiersTER;

import weka.classifiers.bayes.NaiveBayesMultinomial;
import weka.core.Instance;
import weka.core.Instances;

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
		for(int c = 0; c<m_numClasses; c++)
		{
			nbOfClassesContainingWord[c] = 0;
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
	
		
		//TODO



	}


}
