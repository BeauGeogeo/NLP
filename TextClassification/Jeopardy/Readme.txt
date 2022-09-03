*************************************************** Jeopardy - Internship project ***************************************************


						*** Context of the project ***


The project was realized during a one month internship at the Departement des donnéees, Service central de renseignement criminel
(SCRC) de la Gendarmerie Nationale, at Cergy Pontoise. At that time, I had just finished my 1st year of College in Science and
Engineering, which also was the 1st year of professional retraining from law to science and resumption of study.


						*** Aim of the internship and the project ***


The aim of the internship was to discover some basic principles of AI and meet researchers and scientist in this field. Alongside AI
conferences, my tutor made me discover some principles of NLP and the Spacy Library, in order to make text classification. 
I then chose a dataset, cleaned and prepared the data, and adapted code from the internet to perform this task.

I downloaded the Jeopardy dataset as a JSON file. Jeopardy is a quizz game with many questions in many different categories. My goal
was to predict to which category a given question belongs to. The problem was that there were so many categories that even if the
number of questions in the dataset was quite big, each category only contained a few questions. This is problematic regarding the
fact that a supervised algorithm, the kind of algorithm used here to do classsification, requires a considerable amount of examples to
perform well. Fortunately, many of these categories overlapped, so I just had to merge them to have enough data for training. 
So I made two big categories designed to have the biggest training dataset possible to perform binary classification. It resulted in
the following task : given a question, does it belong to the 'HISTORY' category or to the 'SCIENCE' one ?

Although some difficulties have been discarded without impacting too much the performances of the algorithm, they're worth 
mentionning.
First, there is sometimes ambiguity, stemming from the fact that some examples could belong to both category, because you can have
questions about History of science and questions about science containing historical elements. 
Moreover, some categories merged in the science category might not really belong to it even if the name science is in it, such as
occult sciences or religious sciences, regarding that the meaning of the word science here is a little bit different.


						*** Tasks performed by the script ***


First, there is a data preparation and cleaning task, using BeautifulSoup library. 

Then, we train and evaluate the model.

Finally, we scrutinize some predictions made on a bunch of examples. 

