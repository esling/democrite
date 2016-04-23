%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                                                 %
%                                       Code                                                      %
% Dynamic Pooling and Unfolding Recursive Autoencoders for Paraphrase Detection                   %
% Richard Socher, Eric H. Huang, Jeffrey Pennington, Andrew Y. Ng, and Christopher D. Manning     %
% Advances in Neural Information Processing Systems (NIPS 2011)                                   %
% See http://www.socher.org for more information or to ask questions                              %
%                                                                                                 %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

This code outputs whether a given pair of sentences has a paraphrase
relationship based on a trained, unfolding recursive neural network as
described in the above paper.
It is designed to be easy to use, all you need to do
is to put paraphrase label and pairs of sentences into a text file, one label or
sentence per line. The output will be another 
textfile with the model's predictions. 

This code is provided as is. It is free for 
academic, non-commercial purposes. 
For questions, please contact richard @ socher .org



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                   %
%                Installation                       %
%                                                   %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

- The code runs on any linux machine with bash, 
  matlab and java installed.

- After unpacking the zip file go to folder and make 
  sure the executables have permission:

chmod 777 classifyParaphrases.sh
chmod 777 stanford-parser-2011-09-14/lexparser.sh


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                   %
%                Running the Code                   %
%                                                   %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

- You can see how it classifies pairs of sentences 
  in the input.txt file by just running:

  ./classifyParaphrases.sh
  
- To get the model's predictions on your own pairs of sentences, 
  you need to change the file 

  input.txt

- The default content of the input.txt file is the test set of the MSR
  Paraphrase dataset. Running ./classifyParaphrase.sh as provided will
  reproduce the state-of-the-art result in the above paper.

- The input file 'input.txt' should be a pair of sentences and the ground 
  truth labels of whether they are paraphrases (1 if they are, 0 otherwise),
  and should have the following format:
  <label (0 or 1)>
  <sentence 1>
  <sentence 2>
  <label (0 or 1)>
  <sentence 1>
  <sentence 2>
  ...

- The code will then produce as output a text file:

  output.txt  

- In this file, the nth line of the file is the 
  vector for the nth pair in the input.txt file.




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                   %
%                Included Packages                  %
%                                                   %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

This archive includes 2 external packages for convenience:

- The word vectors of the paper "Word representations: 
  A simple and general method for semi-supervised 
  learning" by Turian et al (ACL, 2010).
  These word vectors are available at: 
  http://metaoptimize.com/projects/wordreprs/ 

- The Stanford Parser of the paper "Accurate 
  Unlexicalized Parsing" by Dan Klein and 
  Christopher D. Manning. 2003.
  The parser is available at 
  http://nlp.stanford.edu/software/lex-parser.shtml


  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                   %
%                Bibtex                             %
%                                                   %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  
  If you use the code, please cite:
  
  @incollection{SocherEtAl2011:PoolRAE,
   title = {{Dynamic Pooling and Unfolding Recursive Autoencoders for Paraphrase Detection}},
   author = {Richard Socher and Eric H. Huang and Jeffrey Pennington and Andrew Y. Ng and Christopher D. Manning},
   booktitle = {{Advances in Neural Information Processing Systems 24}},
   year = {2011}
  }
  
