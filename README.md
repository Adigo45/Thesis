Interpreting Customer Churn Prediction : An Explainable AI approach for Enhancing Business Insights.* (use style: paper title)
*Note: Sub-titles are not captured in Xplore and should not be used

 
line 1: 1st Given Name Surname 
line 2: dept. name of organization (of Affiliation)
line 3: name of organization (of Affiliation)
line 4: City, Country
line 5: email address or ORCID

line 1: 2nd Given Name Surname
line 2: dept. name of organization (of Affiliation)
line 3: name of organization (of Affiliation)
line 4: City, Country
line 5: email address or ORCID

line 1: 3rd Given Name Surname
line 2: dept. name of organization (of Affiliation)
line 3: name of organization (of Affiliation)
line 4: City, Country
line 5: email address or ORCID

line 1: 4th Given Name Surname
line 2: dept. name of organization (of Affiliation)
line 3: name of organization (of Affiliation)
line 4: City, Country
line 5: email address or ORCID 

 



line 1: 5th Given Name Surname
line 2: dept. name of organization (of Affiliation)
line 3: name of organization (of Affiliation)
line 4: City, Country
line 5: email address or ORCID
line 1: 6th Given Name Surname
line 2: dept. name of organization (of Affiliation)
line 3: name of organization (of Affiliation)
line 4: City, Country
line 5: email address or ORCID





 

 
Abstract—In today's highly competitive market, retaining customers and fostering business growth has become increasingly challenging. Consequently, the extensive utilization of machine learning models in developing churn prediction models has posed a challenge due to their opaque nature. As a result, understanding the reasoning behind their predictions can take time and effort. This research paper introduces a novel approach incorporating LIME (Local Interpretable Model-Agnostic Explanations) to address this issue to offer local interpretability for individual predictions. Additionally, SHAP (Shapely Additive Explanations) values are employed to quantify the contribution of each feature toward the final prediction. To evaluate the proposed approach, our study employed a real-world dataset obtained from a specific bank containing customer data from the European region. Various machine learning algorithms, such as KNN, SVM, Random Forest, ANN, and XGBoost, were utilized, and their accuracy and performance were compared. However, the primary focus of our paper is on the XGBoost model, known for its time efficiency and higher accuracy.
Keywords— Customer Churn prediction, Explainable AI, XGBoost, Machine learning Models, LIME
Introduction 
Customer churn, which refers to customers discontinuing their use of a product or service, is a significant concern for businesses across various industries, including banking. Understanding this phenomenon is crucial for businesses to take proactive measures in retaining customers, enhancing customer satisfaction, and ultimately boosting revenue. The rise of AI and machine learning has led to a remarkable increase in the use of these technologies for predicting customer churn.
However, one of the main challenges in using AI for customer churn prediction is the "black box" problem, which is not suitable for regulated financial services. Many AI models are complex and difficult to interpret, making it challenging to understand the reasoning behind their predictions. To overcome this limitation, Explainable AI (EXAI) models have emerged. These models provide transparency and interpretability, enabling businesses to comprehend why a model makes specific predictions and take appropriate actions accordingly. EXAI is currently applied in various industries, including sales, human resources, healthcare, disease diagnosis and classification, medicine distribution, and the financial sector.
To create and utilize these AI models, it is essential to understand what explainable AI means. The term was first used by [2] to describe how well a system could explain the actions of AI-controlled characters in simulation games. Although the phrase is relatively new, the need for explainability dates back to the mid-1970s when researchers began exploring explanations for expert systems. However, as AI progressed with the development of machine learning, the focus shifted towards creating models and algorithms that prioritize predictive capabilities, causing the progress in explainability to slow down.
According to the Bank of England [4], explainability means that stakeholders can comprehend the main drivers behind decisions made by model-driven systems. The concept of explainability (also referred to as interpretability) implies that a machine learning model and its output can be communicated to humans in a way that makes sense at an acceptable level. Miller defines interpretability as the extent to which a human can understand the cause of a decision [5]. Achieving interpretability with customers can promote transparency and confidence, allowing banks to reassure clients that decisions are not arbitrary and helping them understand the factors influencing their relationship with the bank, such as why a customer is likely to churn.
Regulated financial services, like the German Federal Financial Supervisory Authority [6], do not accept "black box excuses," emphasizing the importance of providing explanations. The General Data Protection Regulation (GDPR) of the European Union also requires businesses to use personal data for automated processing to explain how the system makes choices. Furthermore, the GDPR grants individuals in the European Union a "right to explanation" regarding automated decision-making. Hence, providing explanations for model outputs is essential, and advancements in understanding these systems can lead to wider acceptance of novel approaches in financial applications.
Linear and logistic regression models are considered more interpretable but have lower accuracy compared to artificial neural networks and tree models, which are deemed more accurate. To bridge this gap, we will utilize the XGBoost model in our study. XGBoost is highly precise and time-efficient and incorporates a cutting-edge technique for interpreting its final predictions. Additionally, we will evaluate various machine learning models and neural networks alongside our model. Our research goes beyond prediction and focuses on explaining the model's decisions.
Although Rahman's model [8] has shown high accuracy in predicting customer churn in banks, its lack of interpretability makes it challenging for bank management to understand the underlying causes of customer turnover. Caparrini [9] explores the explainability of machine learning models, providing grading for peer-to-peer models with a focus on Shapely Additive Explanations (SHAP). However, our study aims to investigate the usage of SHAP and Local Interpretable Model-Agnostic Explanations (LIME) [10] to offer more precise and understandable explanations of model predictions 
The rest of the paper discusses about the related literature of credit scoring and reinforcement learning in section 2. In Section 3 the methodology presented in this paper is discussed followed by experimental results in section 4. Finally, conclusions from the work reported here are summarized in section 5 along with future directions.

Related work
Statistical Logistic Regression Model

A statistical logistic regression model is a type of regression analysis used to predict the probability of a binary outcome. It is commonly employed when the dependent variable, the response variable, has two possible outcomes (e.g., yes/no, success/failure). Logistic regression aims to estimate the relationship between the independent variables (also called predictors or explanatory variables) and the probability of a specific outcome occurring. And because our customer churn model relies on two assumptions to determine if a customer will leave, it can be used. These models have now been adopted by the scientific community in the fields of economics, finance, and other social and environmental sciences [12] [13]. 

Create a response variable, Yn, for each customer, n, to indicate whether the client has left the company or not; Yn is equal to 1 if the customer has left and 0 otherwise. Additionally, let Xn represent a vector of explanatory variables. A collection of predictors is employed in the Logistic Regression model to calculate the probability that an event will occur. Here is what the logistic regression predicts will happen.
ln⁡(p_n/(1-p_n ))  = α+ ∑_(j=1)^J▒〖β_j x_nj 〗
Is a linear representation of the input variables in the equation, taking a value between -∞ and +∞, while taking a value between 0 and 1. Numerous statistical flaws exist in LR. Two of these are multi-collinearity and decreased performance precision.

Thammadi and Gangadharaiah (2019), in their study, focus on predicting customer churn in the banking industry using logistic regression and decision trees. It highlights the importance of logistic regression as a reliable and interpretable model for customer churn prediction.Deng et al. (2015) explore the use of deep learning techniques in customer churn prediction and highlight logistic regression's significance as a baseline model. In addition, it discusses the interpretability of logistic regression and its role in understanding the drivers of customer churn. Verbeke and  Martens (2012) compare logistic regression performance with other classification techniques for customer churn prediction in the banking industry. It demonstrates the effectiveness of logistic regression and provides insights into its application in predicting customer churn. Ngai et al. (2009), in their work, provide a comprehensive literature review on the application of data mining techniques, including logistic regression, in customer relationship management. It discusses the use of logistic regression for customer churn prediction and highlights its advantages and limitations.

Application of Machine Learning for customer Churn
Use either SI (MKS) or CGS as primary units. (SI units are encouraged.) English units may be used as secondary units (in parentheses). An exception would be the use of English units as identifiers in trade, such as “3.5-inch disk drive”.
Avoid combining SI and CGS units, such as current in amperes and magnetic field in oersteds. This often leads to confusion because equations do not balance dimensionally. If you must use mixed units, clearly state the units for each quantity that you use in an equation.
Do not mix complete spellings and abbreviations of units: “Wb/m2” or “webers per square meter”, not “webers/m2”.  Spell out units when they appear in text: “. . . a few henries”, not “. . . a few H”.
Use a zero before decimal points: “0.25”, not “.25”. Use “cm3”, not “cc”. (bullet list)
Equations
The equations are an exception to the prescribed specifications of this template. You will need to determine whether or not your equation should be typed using either the Times New Roman or the Symbol font (please no other font). To create multileveled equations, it may be necessary to treat the equation as a graphic and insert it into the text after your paper is styled.
Number equations consecutively. Equation numbers, within parentheses, are to position flush right, as in (1), using a right tab stop. To make your equations more compact, you may use the solidus ( / ), the exp function, or appropriate exponents. Italicize Roman symbols for quantities and variables, but not Greek symbols. Use a long dash rather than a hyphen for a minus sign. Punctuate equations with commas or periods when they are part of a sentence, as in:
	ab	
Note that the equation is centered using a center tab stop. Be sure that the symbols in your equation have been defined before or immediately following the equation. Use “(1)”, not “Eq. (1)” or “equation (1)”, except at the beginning of a sentence: “Equation (1) is . . .”
Some Common Mistakes
The word “data” is plural, not singular.
The subscript for the permeability of vacuum 0, and other common scientific constants, is zero with subscript formatting, not a lowercase letter “o”.
In American English, commas, semicolons, periods, question and exclamation marks are located within quotation marks only when a complete thought or name is cited, such as a title or full quotation. When quotation marks are used, instead of a bold or italic typeface, to highlight a word or phrase, punctuation should appear outside of the quotation marks. A parenthetical phrase or statement at the end of a sentence is punctuated outside of the closing parenthesis (like this). (A parenthetical sentence is punctuated within the parentheses.)
A graph within a graph is an “inset”, not an “insert”. The word alternatively is preferred to the word “alternately” (unless you really mean something that alternates).
Do not use the word “essentially” to mean “approximately” or “effectively”.
In your paper title, if the words “that uses” can accurately replace the word “using”, capitalize the “u”; if not, keep using lower-cased.
Be aware of the different meanings of the homophones “affect” and “effect”, “complement” and “compliment”, “discreet” and “discrete”, “principal” and “principle”.
Do not confuse “imply” and “infer”.
The prefix “non” is not a word; it should be joined to the word it modifies, usually without a hyphen.
There is no period after the “et” in the Latin abbreviation “et al.”.
The abbreviation “i.e.” means “that is”, and the abbreviation “e.g.” means “for example”.
An excellent style manual for science writers is [7].
Using the Template
After the text edit has been completed, the paper is ready for the template. Duplicate the template file by using the Save As command, and use the naming convention prescribed by your conference for the name of your paper. In this newly created file, highlight all of the contents and import your prepared text file. You are now ready to style your paper; use the scroll down window on the left of the MS Word Formatting toolbar.
Authors and Affiliations
The template is designed for, but not limited to, six authors. A minimum of one author is required for all conference articles. Author names should be listed starting from left to right and then moving down to the next line. This is the author sequence that will be used in future citations and by indexing services. Names should not be listed in columns nor group by affiliation. Please keep your affiliations as succinct as possible (for example, do not differentiate among departments of the same organization).
For papers with more than six authors: Add author names horizontally, moving to a third row if needed for more than 8 authors.
For papers with less than six authors: To change the default, adjust the template as follows.
Selection: Highlight all author and affiliation lines.
Change number of columns: Select the Columns icon from the MS Word Standard toolbar and then select the correct number of columns from the selection palette.
Deletion: Delete the author and affiliation lines for the extra authors.

Identify the Headings
Headings, or heads, are organizational devices that guide the reader through your paper. There are two types: component heads and text heads.
Component heads identify the different components of your paper and are not topically subordinate to each other. Examples include Acknowledgments and References and, for these, the correct style to use is “Heading 5”. Use “figure caption” for your Figure captions, and “table head” for your table title. Run-in heads, such as “Abstract”, will require you to apply a style (in this case, italic) in addition to the style provided by the drop down menu to differentiate the head from the text.
Text heads organize the topics on a relational, hierarchical basis. For example, the paper title is the primary text head because all subsequent material relates and elaborates on this one topic. If there are two or more sub-topics, the next level head (uppercase Roman numerals) should be used and, conversely, if there are not at least two sub-topics, then no subheads should be introduced. Styles named “Heading 1”, “Heading 2”, “Heading 3”, and “Heading 4” are prescribed.
Figures and Tables
 Positioning Figures and Tables: Place figures and tables at the top and bottom of columns. Avoid placing them in the middle of columns. Large figures and tables may span across both columns. Figure captions should be below the figures; table heads should appear above the tables. Insert figures and tables after they are cited in the text. Use the abbreviation “Fig. 1”, even at the beginning of a sentence.
Table Type Styles
Table Head	Table Column Head
	Table column subhead	Subhead	Subhead
copy	More table copya		
Sample of a Table footnote. (Table footnote)
Example of a figure caption. (figure caption)
Figure Labels: Use 8 point Times New Roman for Figure labels. Use words rather than symbols or abbreviations when writing Figure axis labels to avoid confusing the reader. As an example, write the quantity “Magnetization”, or “Magnetization, M”, not just “M”. If including units in the label, present them within parentheses. Do not label axes only with units. In the example, write “Magnetization (A/m)” or “Magnetization {A[m(1)]}”, not just “A/m”. Do not label axes with a ratio of quantities and units. For example, write “Temperature (K)”, not “Temperature/K”.
Acknowledgment (Heading 5)
The preferred spelling of the word “acknowledgment” in America is without an “e” after the “g”. Avoid the stilted expression “one of us (R. B. G.) thanks ...”.  Instead, try “R. B. G. thanks...”. Put sponsor acknowledgments in the unnumbered footnote on the first page. 
References
The template will number citations consecutively within brackets [1]. The sentence punctuation follows the bracket [2]. Refer simply to the reference number, as in [3]—do not use “Ref. [3]” or “reference [3]” except at the beginning of a sentence: “Reference [3] was the first ...”
Number footnotes separately in superscripts. Place the actual footnote at the bottom of the column in which it was cited. Do not put footnotes in the abstract or reference list. Use letters for table footnotes.
Unless there are six authors or more give all authors’ names; do not use “et al.”. Papers that have not been published, even if they have been submitted for publication, should be cited as “unpublished” [4]. Papers that have been accepted for publication should be cited as “in press” [5]. Capitalize only the first word in a paper title, except for proper nouns and element symbols.
For papers published in translation journals, please give the English citation first, followed by the original foreign-language citation [6].
[1] Saraswat, D., Bhattacharya, P., Verma, A., Prasad, V. K., Tanwar, S., Sharma, G., ... & Sharma, R. (2022). Explainable AI for healthcare 5.0: opportunities and challenges. IEEE Access.
[2] Van Lent, M., Fisher, W., & Mancuso, M. (2004, July). An explainable artificial intelligence system for small-unit tactical behavior. In Proceedings of the national conference on artificial intelligence (pp. 900-907). Menlo Park, CA; Cambridge, MA; London; AAAI Press; MIT Press; 1999.
[3] Moore, J. D., & Swartout, W. R. (1988). Explanation in expert systemss: A survey. University of Southern California Marina del Rey Information Sciences Inst.
[4] Joseph, A., & Regressions, S. (2019). A framework for statistical inference on machine learning models. Working Paper, Bank of England.
[5] Miller, T. (2019). Explanation in artificial intelligence: Insights from the social sciences. Artificial intelligence, 267, 1-38.
[6] Authority, F. F. S. (2018). Big Data Meets Artificial Intelligence: Challenges and Implications for the Supervision and Regulation of Financial Services. Federal Financial Supervision Authority.
[7] Voss, W. G. (2016). European union data privacy law reform: General data protection regulation, privacy shield, and the right to delisting. The Business Lawyer, 72(1), 221-234.
[8] Rahman, M., & Kumar, V. (2020, November). Machine learning based customer churn prediction in banking. In 2020 4th international conference on electronics, communication and aerospace technology (ICECA) (pp. 1196-1201). IEEE.
[9] Ariza-Garzón, M. J., Arroyo, J., Caparrini, A., & Segovia-Vargas, M. J. (2020). Explainability of a machine learning granting scoring model in peer-to-peer lending. Ieee Access, 8, 64873-64890.
[10] Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. Advances in neural information processing systems, 30.
[11] "Scikit learn documentation," [Online]. Available: https://scikit-learn.org/stable/modules/linear_model.html. [Accessed 04 2023].
[12] Jabeur, S. B. (2017). Bankruptcy prediction using partial least squares logistic regression. Journal of Retailing and Consumer Services, 36, 197-202.
[13] He, Y., He, Z., & Zhang, D. (2009, August). A study on prediction of customer churn in fixed communication network based on data mining. In 2009 sixth international conference on fuzzy systems and knowledge discovery (Vol. 1, pp. 92-94). IEEE.
[14] Murdoch, W. J., Singh, C., Kumbier, K., Abbasi-Asl, R., & Yu, B. (2019). Definitions, methods, and applications in interpretable machine learning. Proceedings of the National Academy of Sciences, 116(44), 22071-22080.
[15] Chen, T., & Guestrin, C. (2016, August). Xgboost: A scalable tree boosting system. In Proceedings of the 22nd acm sigkdd international conference on knowledge discovery and data mining (pp. 785-794).
[16] Mai, F., Tian, S., Lee, C., & Ma, L. (2019). Deep learning models for bankruptcy prediction using textual disclosures. European journal of operational research, 274(2), 743-758.
[17] A. Mello, "Towards Data Science," 17 August 2020. [Online]. Available: https://towardsdatascience.com/xgboost-theory-and-practice-fb8912930ad6. [Accessed 27 April 2023].
[18] Guliyev, H., & Tatoğlu, F. Y. (2021). Customer churn analysis in banking sector: Evidence from explainable machine learning models. Journal Of Applied Microeconometrics, 1(2), 85-99.
[19] U. Jaitley, "Towards Data Science," 2 feb 2019. [Online]. Available: https://towardsdatascience.com/comparing-different-classification-machine-learning-models-for-an-imbalanced-dataset-fdae1af3677f. [Accessed 28 04 2023].
[20] Ahmad, A. K., Jafar, A., & Aljoumaa, K. (2019). Customer churn prediction in telecom using machine learning in big data platform. Journal of Big Data, 6(1), 1-24.
[21] Ribeiro, M. T., Singh, S., & Guestrin, C. (2016, August). " Why should i trust you?" Explaining the predictions of any classifier. In Proceedings of the 22nd ACM SIGKDD international conference on knowledge discovery and data mining (pp. 1135-1144).

IEEE conference templates contain guidance text for composing and formatting conference papers. Please ensure that all template text is removed from your conference paper prior to submission to the conference. Failure to remove template text from your paper may result in your paper not being published. 

