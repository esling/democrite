Stanford Lexicalized Parser v1.6.9 - 2011-09-14
------------------------------------------------

Copyright (c) 2002-2011 The Board of Trustees of The Leland Stanford Junior
University. All Rights Reserved.

Original core parser code by Dan Klein.  Support code, additional modules,
languages, features, internationalization, compaction, typed dependencies,
etc. by Christopher Manning, Roger Levy, Teg Grenager, Galen Andrew,
Marie-Catherine de Marneffe, Jenny Finkel, Spence Green, Bill MacCartney, Anna
Rafferty, Huihsin Tseng, Pi-Chuan Chang, Wolfgang Maier, and Richard Eckart.

This release prepared by John Bauer.

This package contains 3 parsers: a high-accuracy unlexicalized PCFG, a
lexicalized dependency parser, and a factored model, where the estimates
of dependencies and an unlexicalized PCFG are jointly optimized to give a
lexicalized PCFG treebank parser.  Also included are grammars for various
languages for use with these parsers.

For more information about the parser API, point a web browser at the
included javadoc directory (use the browser's Open File command to open
the index.html file inside the javadoc folder).  Start by looking at the
Package page for the edu.stanford.nlp.parser.lexparser package, and then
look at the page for the LexicalizedParser class documentation therein,
particularly documentation of the main method.

Secondly, you should also look at the Parser FAQ on the web:

    http://nlp.stanford.edu/software/parser-faq.shtml

This software requires Java 5 (JDK 1.5.0+).  (You must have installed it
separately. Check that the command "java -version" works and gives 1.5+.)


QUICKSTART

UNIX COMMAND-LINE USAGE

On a Unix system you should be able to parse the English test file with the
following command:

    ./lexparser.sh data/testsent.txt

This uses the PCFG parser, which is quick to load and run, and quite accurate.

[Notes: it takes a few seconds to load the parser data before parsing
begins; continued parsing is quicker. To use the lexicalized parser, replace
englishPCFG.ser.gz with englishFactored.ser.gz in the lexparser.sh script
and use the flag -mx600m to give more memory to java.]

WINDOWS GUI USAGE

On a Windows system, assuming that java is on your PATH, you should be able
to run a parsing GUI by double-clicking on the lexparser-gui.bat icon,
or giving the command lexparser-gui in this directory from a command prompt.

Click Load File, Browse, and navigate to and select testsent.txt in the
top directory of the parser distribution.  Click Load Parser, Browse, and
select englishPCFG.ser.gz in the same directory.  Click Parse to parse the
first sentence.

OTHER USE CASES

The GUI is also available under Unix:

    lexparser-gui.sh

Under Mac OS X, you can double-click on lexparser-gui.command to invoke the
GUI.  The command-line version works on all platforms.	Use lexparser.bat
to run it under Windows.  The GUI is only for exploring the parser. It does
not allow you to save output.  You need to use the command-line program or
programmatic API to do serious work with the parser.

ADDITIONAL GRAMMARS

The parser is supplied with several trained grammars. There are English
grammars based on the standard LDC Penn Treebank WSJ training sections 2-21
(wsj*), and ones based on an augmented data set, better for questions,
commands, and recent English and biomedical text (english*).

All grammars are located in the /grammar directory.

MULTILINGUAL PARSING
In addition to the English grammars, the parser comes with trained grammars
for Arabic, Chinese, French, and German. To parse with these grammars, run

    lexparser-lang.sh

with no arguments to see usage instructions. You can change language-specific
settings passed to the parser by modifying lexparser_lang.def.

You can also train and evaluate new grammars using:

    lexparser-lang-train-test.sh

To see how we trained the grammars supplied in this distribution, see

    bin/makeSerialized.csh

You will not be able to run this script (since it uses Stanford-specific file
paths), but you should be able to see what we did.

To setup the classpath for the multilingual parsing scripts, first run:

    ./install.sh

This will create the file classpath.def. You can check to ensure that
this file contains the correct path to stanford-parser.jar.

Arabic
Trained on parts 1-3 of the Penn Arabic Treebank (ATB) using the
pre-processing described in (Green and Manning, 2010). The default input
encoding is UTF-8 Arabic script.  To parse with Buckwalter encoding, we
recommend conversion to UTF-8 using the package

    edu.stanford.nlp.international.arabic.Buckwalter

which is included in stanford-parser.jar.

Note that the parser *requires* clitic segmentation per the ATB standard
prior to parsing. A freely available package for performing this segmentation
is MADA+TOKAN:

    http://www1.cs.columbia.edu/~rambow/software-downloads/MADA_Distribution.html

Chinese
There are Chinese grammars trained just on mainland material from
Xinhua and more mixed material from the LDC Chinese Treebank. The default
input encoding is GB18030.

French
Trained on the functionally annotated section of the French Treebank
(FTB) using the pre-processing described in (Green et al., 2011). Tokenization
according to the FTB standard is required prior to parsing. A freely available
set of tokenization tools can be found at:

    http://gforge.inria.fr/projects/lingwb/

German
Trained on the Negra corpus. Details are included in (Rafferty and
Manning, 2008).

TREEBANK PREPROCESSING

The pre-processed versions of the ATB described
in (Green and Manning, 2010) and the FTB described in (Green et al.,
2011) can be reproduced using the TreebankPreprocessor included in this
release. The configuration files are located in /conf. For example,
to create the ATB data, run:

    bin/run-tb-preproc -v conf/atb-latest.conf

Note that you'll need to update the conf file paths to your local treebank
distributions as the data is not distributed with the parser. You'll
also need to set the classpath in the cmd_line variable of run-tb-preproc.

The TreebankPreprocessor conf files support various options, which are
documented in

    edu.stanford.nlp.international.process.ConfigParser

EVALUATION METRICS

The Stanford parser comes with Java implementations of the following
evaluation metrics:

    Dependency Labeled Attachment

    Evalb         (Collins, 1997)
      -Includes per-category evaluation with the -c option

    Leaf Ancestor (Sampson and Babarczy, 2003)
      -Both micro- and macro-averaged score

    Tagging Accuracy

See the usage instructions and javadocs in the requisite classes located in
edu.stanford.nlp.parser.metrics.

LICENSE

// StanfordLexicalizedParser -- a probabilistic lexicalized NL CFG parser
// Copyright (c) 2002-2010 The Board of Trustees of
// The Leland Stanford Junior University. All Rights Reserved.
//
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License
// as published by the Free Software Foundation; either version 2
// of the License, or (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.
//
// For more information, bug reports, fixes, contact:
//    Christopher Manning
//    Dept of Computer Science, Gates 1A
//    Stanford CA 94305-9010
//    USA
//    parser-support@lists.stanford.edu
//    http://nlp.stanford.edu/downloads/lex-parser.shtml


CHANGES

This section summarizes changes between released versions of the parser.

Version 1.6.9  2011-09-14
   Added some imperatives to the English training data; added root dependency.
Version 1.6.8  2011-06-15
   Added French parser and leaf ancestor evaluation metric;
   reorganized distribution; new data preparation scripts;
   rebuilt grammar models; other bug fixes
Version 1.6.7  2011-05-15
    Minor bug fixes
Version 1.6.6  2011-04-17
    Compatible with tagger, corenlp and tregex.
Version 1.6.5  2010-10-30
    Further improvements to English Stanford Dependencies and other minor
    changes
Version 1.6.4  2010-08-16
    More minor bug fixes and improvements to English Stanford Dependencies
    and question parsing
Version 1.6.3  2010-07-09
    Improvements to English Stanford Dependencies and question parsing,
    minor bug fixes
Version 1.6.2  2010-02-25
    Improvements to Arabic parser models, and to English and Chinese Stanford
    Dependencies
Version 1.6.1  2008-10-19
    Slightly improved Arabic, German and Stanford Dependencies
Version 1.6  2007-08-18
    Added Arabic, k-best PCCFG parsing; improved English grammatical relations
Version 1.5.1  2006-05-30
    Improved English and Chinese grammatical relations; fixed UTF-8 handling
Version 1.5  2005-07-20
    Added grammatical relations output; fixed bugs introduced in 1.4
Version 1.4  2004-03-24
    Made PCFG faster again (by FSA minimization); added German support
Version 1.3  2003-09-06
    Made parser over twice as fast; added tokenization options
Version 1.2  2003-07-20
    Halved PCFG memory usage; added support for Chinese
Version 1.1  2003-03-25
    Improved parsing speed; included GUI, improved PCFG grammar
Version 1.0  2002-12-05
    Initial release
