//package classifiersTER;
package weka.classifiers.bayes;

import weka.core.Instance;

public class CFCTERab extends NaiveBayesMultinomialTERab {

	/**
	 * 
	 */
	private static final long serialVersionUID = -3765857581531315570L;


	public double [] distributionForInstance(Instance instance) throws Exception 
	{
		double[] probOfClassGivenDoc = new double[m_numClasses];
		double min = 1;
		
		for(int i = 0; i<m_numClasses; i++) 
		{
			probOfClassGivenDoc[i] =  probOfDocGivenClass(instance, i);
		}


		return probOfClassGivenDoc;
	}
	private double probOfDocGivenClass(Instance inst, int classIndex)
	{
		//On commence par calculer le vecteur document
		double answer;
		double[] doc = new double[m_numAttributes+1];
		
		for(int a = 0; a<inst.numValues(); a++)
			if(inst.index(a) != inst.classIndex())
			{
				if(!inst.isMissing(a) && inst.valueSparse(a)!=0)
				{

					doc[inst.index(a)] = inst.valueSparse(a); //peut etre qu'on peut mettre 1
					
				}
			}
		doc[m_numAttributes]=1;//On ajoute une dimension pour eviter des vecteurs nuls
		
		double[] vClass = new double[m_numAttributes+1]; 
		for(int i =0; i< m_numAttributes;i++)
			vClass[i] = m_probOfWordGivenClass[classIndex][i];
		vClass[m_numAttributes]=1;
		//System.out.println(":::"+calculSimilariteCosinus(doc, vClass));
		
		return calculSimilariteCosinus(doc, vClass);
	}
	//On suppose que les 2 vecteur ont la meme dimension, et que les 2 ne sont pas nuls
	//Ce serai pas mal de lancer une exception, si on a le temps
	private double calculSimilariteCosinus(double[] v1, double[] v2){
		double ps = 0,norm1=0,norm2=0;//On initialise le ps a 1, pour eviter toute valeur nulle
		for(int i =0;i< v1.length;i++){
			ps += v1[i]*v2[i];
			norm1 += v1[i]*v1[i];
			norm2 += v2[i]*v2[i];
		}
		norm1 = Math.sqrt(norm1);
		norm2 = Math.sqrt(norm2);
		return ps / (norm1*norm2);//Pas besoin de passer par arccos, on obtient déja un nombre entre 0 et 1, car les vecteur n'ont que des valeurs positives ou nulles
	}
	

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		runClassifier(new CFCTERab(), args);

	}

}
