% µthesis
% 
% Based on ociamthesis
% Originally by Keith A. Gillow (gillow@maths.ox.ac.uk), 1997
% Modified by Sam Evans (sam@samuelevansresearch.org), 2007
% Modified by John McManigle (john@oxfordechoes.com), 2015
%
% Copyright (c) 2025 Daniel Dorfmeister
%
% Broad permissions are granted to use, modify, and distribute this software
% as specified in the MIT License included in this distribution's LICENSE file.
%
%
%-------------------------- identification ---------------------
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{µthesis}[2025/07/18 µCSRL PhD thesis class]
%-------------------------- initial code -----------------------

% JEM: Defaults to one-side, but passing 'twoside' should set everything up for that
\LoadClass[openright,10pt]{report}

\newif\ifmc@nobind
\mc@nobindfalse

\DeclareOption{nobind}{\mc@nobindtrue}
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{report}}
\ProcessOptions\relax

\setlength{\parskip}{0pt plus 1pt}
\setlength{\parfillskip}{0pt plus .7\textwidth}

% JEM: Lengths for single spacing (ie separate abstract, captions), front matter (abstract,
%   acknowledgements, table of contents, etc), and main body text.
\newlength{\singlebaselineskip}
\newlength{\frontmatterbaselineskip}
\newlength{\textbaselineskip}

\setlength{\singlebaselineskip}{\baselineskip}
\setlength{\frontmatterbaselineskip}{15pt plus2pt minus1pt}
\setlength{\textbaselineskip}{15pt plus2pt minus1pt}

\usepackage{varwidth}
\newcommand{\originalitytext}{%
   \rule[-2mm]{1pt}{7mm}\hspace{-1pt}%
   \rule[-2mm]{7mm}{1pt}\hspace{-1pt}%
   \rule[-2mm]{1pt}{7mm}\hspace{-7mm}%
   \rule[5mm]{7mm}{1pt}\hspace{1em}%
   \begin{varwidth}{\textwidth}I hereby certify that this is entirely\\my own work unless otherwise stated.\end{varwidth}}

%
% DECLARATIONS
%
% These macros are used to declare arguments needed for the
% construction of the title page and other preamble.

% The year and term the degree will be officially conferred
\def\submissiondate#1{\gdef\@submissiondate{#1}}
\def\@submissiondate{}
\def\acceptancedate#1{\gdef\@acceptancedate{#1}}
\def\examinationdate#1{\gdef\@examinationdate{#1}}
% The names of the examiners
\def\firstexaminer#1{\gdef\@firstexaminer{#1}}
\def\secondexaminer#1{\gdef\@secondexaminer{#1}}
% The full (unabbreviated) name of the degree
\def\degree#1{\gdef\@degree{#1}}
% The name of your department
\def\department#1{\gdef\@department{#1}}
% The language of the title page, options: de (default), en
\def\titlelang#1{\gdef\@titlelang{#1}}

%
% CRESTS
%

\def\crest{{\includegraphics{figures/unibw.pdf}}}


%
% Define text area of page and margin offsets
%

\ifmc@nobind
\usepackage[includehead,hmargin={3.1cm, 3.1cm}, vmargin={2.5cm,2.7cm}, headsep=.8cm,footskip=1.2cm]{geometry}
\else
\usepackage[includehead,hmargin={3.6cm, 2.6cm}, vmargin={2.5cm,2.7cm}, headsep=.8cm,footskip=1.2cm]{geometry}
\fi

\usepackage{graphicx}

\usepackage{fancyhdr}
\setlength{\headheight}{15pt}
\fancyhf{} % clear the header and footers
\pagestyle{fancy}
\renewcommand{\chaptermark}[1]{\markboth{\thechapter. #1}{\thechapter. #1}}
\renewcommand{\sectionmark}[1]{\markright{\thesection. #1}} 
\renewcommand{\headrulewidth}{0pt}
\fancyhead[LO]{\emph{\leftmark}} 
\fancyhead[RE]{\emph{\rightmark}} 
\fancyhead[RO,LE]{\emph{\thepage}}

\fancypagestyle{plain}{\fancyhf{}\fancyfoot[C]{\emph{\thepage}}}

% JEM fix header on cleared pages for openright
\def\cleardoublepage{\clearpage\if@twoside \ifodd\c@page\else
   \hbox{}
   \fancyhead[RE,LO]{}
   \newpage
   \if@twocolumn\hbox{}\newpage
   \fi
   \fancyhead[LO]{\emph{\leftmark}} 
   \fancyhead[RE]{\emph{\rightmark}} 
   \fi\fi}


\usepackage{microtype}
\usepackage{nicefrac}
\usepackage{textcomp}

\usepackage{longtable} 
	% Allows tables to span multiple pages (this package must be called before hyperref)

\usepackage[font=small,labelfont=bf]{caption}
\usepackage{array}
\captionsetup{format=hang}
\newcolumntype{C}{>{\centering\arraybackslash}X}
\newcolumntype{L}{>{\raggedright\arraybackslash}X}
\newcolumntype{R}{>{\raggedleft\arraybackslash}X}
	% Nicer captions

\usepackage{multicol,multirow,array} 
	% Used to make multiple columns for the indices and for creating columns that span multiple rows in tables

\usepackage{rotating} 
	% To allow tables in landscape mode

\usepackage{booktabs} 
	% For better looking tables

\usepackage{pdfpages} 
	% Allows multi-page pdfs to be inserted as graphics

\usepackage{xfrac}

% JEM the following package allows changing spacing in lists (for acknowledgements)
\usepackage[inline]{enumitem}

% JEM the following allows table columns aligned by decimal point
\usepackage{dcolumn}
\newcolumntype{d}[3]{D{.}{\cdot}{#1} }

\usepackage{appendix} 
	% For helping format appendicies
\usepackage{notoccite} 
	% To format the bibliography	


\usepackage[pdfpagelabels,hidelinks,pdfusetitle]{hyperref}
	% for linking between references, figures, TOC, etc in the pdf document

\usepackage[nohints,tight]{minitoc} 
        \renewcommand{\mtcSfont}{\small}
	\setcounter{minitocdepth}{2} 
	% Generates mini tables of contents per chapter
	
% JEM: The following fixes minitoc to not be multiple-spaced.
\let\oldmtc@verse\mtc@verse
\renewcommand{\mtc@verse}[1]{\oldmtc@verse{#1}\setlength{\baselineskip}{\z@}}

%
% Environments
%

% This macro define an environment for front matter that is always 
% single column even in a double-column document.

\newenvironment{alwayssingle}{%
       \@restonecolfalse
       \if@twocolumn\@restonecoltrue\onecolumn
       \else\if@openright\cleardoublepage\else\clearpage\fi
       \fi}%
       {\if@restonecol\twocolumn
       \else\newpage\thispagestyle{empty}\fi}

% TITLE PAGE
%
% Defines title page layout according to "Promotionsordnung 2023":
% https://www.rz.unibw-muenchen.de/studium/pruefungsamt/formulare/copy_of_formulare-1

\renewcommand{\maketitle}{%
\begin{alwayssingle}\null
  \renewcommand{\footnotesize}{\small}
  \renewcommand{\footnoterule}{\relax}
  \thispagestyle{empty}
  \begin{center}
    {\LARGE\bfseries \@title \par}
    \vspace*{16ex}
    \crest \par
    \vspace*{16ex}
    {\Large \@author \par}
    \vspace*{16ex}
    Vollständiger Abdruck der von der Fakultät für \@department{} der Universität der Bundeswehr München zur Erlangung des akademischen Grades eines \par
    \vspace*{2ex}
    {\large\it \@degree \par}
    \vspace*{2ex}
    angenommenen Dissertation. \par
    \vspace*{8ex}
    {\large Gutachter: \par
      1. \@firstexaminer \par
      2. \@secondexaminer \par}
    \vspace*{8ex}
    \ifx\@submissiondate\@empty\else
    {\small Die Dissertation wurde am \@submissiondate{} bei der Universität der Bundeswehr München \par
    eingereicht und durch die Fakultät für \@department{} am \@acceptancedate{} angenommen. \par
    Die mündliche Prüfung fand am \@examinationdate{} statt.}
    \fi
  \end{center}
  \vfill\null
\end{alwayssingle}}


% DEDICATION
%
% The dedication environment makes sure the dedication gets its
% own page and is set out in verse format.

\newenvironment{dedication}
{\begin{alwayssingle}
  \thispagestyle{empty}
  \vspace*{\fill} \begin{center}}
{\end{center}\vspace*{\fill}\vspace*{\fill} \end{alwayssingle}}


% ACKNOWLEDGEMENTS
%
% The acknowledgements environment puts a large, bold, centered 
% "Acknowledgements" label at the top of the page. The acknowledgements
% themselves appear in a quote environment, i.e. tabbed in at both sides, and
% on its own page.

\newenvironment{acknowledgements}%
{   \begin{alwayssingle}\chapter*{Acknowledgements}
    \thispagestyle{empty}
	\pagestyle{empty}
	\setlength{\baselineskip}{\frontmatterbaselineskip}
}
{\end{alwayssingle}}


%ABSTRACT
%
%The abstract environment puts a large, bold, centered "Abstract" label at
%the top of the page. The abstract itself appears in a quote environment,
%i.e. tabbed in at both sides, and on its own page.

\renewenvironment{abstract}%
{   \begin{alwayssingle}\chapter*{\abstractname}
    \thispagestyle{empty}
    \pagestyle{empty}
   \setlength{\baselineskip}{\frontmatterbaselineskip}}
{\end{alwayssingle}}


%ROMANPAGES
%
% The romanpages environment set the page numbering to lowercase roman one
% for the contents and figures lists. It also resets
% page-numbering for the remainder of the dissertation (arabic, starting at 1).
%
% Edited by JEM

\newenvironment{romanpages}
{\cleardoublepage\setlength{\baselineskip}{\frontmatterbaselineskip}\setcounter{page}{1}\renewcommand{\thepage}{\roman{page}}}
{\cleardoublepage\setcounter{page}{1}\renewcommand{\thepage}{\arabic{page}}}


%Figure placement on page
%
%This is to help prevent too many figures on their own pages.  I got it from:
% http://dcwww.camd.dtu.dk/~schiotz/comp/LatexTips/LatexTips.html#captfont

\renewcommand{\topfraction}{0.85}
\renewcommand{\textfraction}{0.1}
\renewcommand{\floatpagefraction}{0.75}


% JEM: VERSE, QUOTATION, and QUOTE must be single-spaced.
% These rename the internal commands for the \begin{...} and
% \end{...} environment commands.
\let\@ldquote=\quote
\let\@ldendquote=\endquote
\renewenvironment{quote}
   {\@ldquote\setlength{\baselineskip}{\singlebaselineskip}}
   {\@ldendquote}

\let\@ldqttn=\quotation
\let\@ldendqttn=\endquotation
\renewenvironment{quotation}
   {\@ldqttn\setlength{\baselineskip}{\singlebaselineskip}}
   {\@ldendqttn}

\let\@ldverse=\verse
\let\@ldendverse=\endverse
\renewenvironment{verse}
   {\@ldverse\setlength{\baselineskip}{\singlebaselineskip}}
   {\@ldendverse}

% JEM: Stuff for List of Abbreviations (or Glossary or other list)
\usepackage{calc}
\newcommand{\t@cfill}{\cleaders\hbox{$\m@th \mkern\@dotsep mu . \mkern\@dotsep mu$}\hfill}
\newcommand{\abbrl@bel}[1]{\makebox[\labelwidth][l]{\textbf{#1}\ \t@cfill}}
\newenvironment{mclistof}[2]{%
   \chapter*{#1}%
   \addcontentsline{toc}{chapter}{#1}
   \begin{list}{}%
   {\renewcommand{\makelabel}{\abbrl@bel}%
    \setlength{\labelwidth}{#2}%
    \setlength{\leftmargin}{\labelwidth+\labelsep}%
    \setlength{\itemsep}{0pt}%
    \markboth{#1}{#1}%
   }}
   {\end{list}\mtcaddchapter}


\usepackage[grey,utopia]{quotchap}
	% To put nice quotes at the beginning of the chapters

\usepackage[nottoc]{tocbibind}
	% Used for inserting the Bibliography, List of Figures, indices, etc into the table of contents.

% JEM: Allow flexible spacing after figures, to reduce other stretching
\setlength{\textfloatsep}{20pt plus15pt minus4pt}

% JEM: Fix paragraph spacing within quotes for multi-par quotes
\renewenvironment{savequote}[1][10cm]{%
\begin{lrbox}{\@quotebox}
\begin{minipage}[t]{#1}\footnotesize\slshape%
\setlength{\parskip}{6pt}}{%
\unskip\end{minipage}\end{lrbox}
\global\setbox\@quotebox\copy\@quotebox
\global\let\@printcites\@iprintcites
\ignorespacesafterend}
\renewcommand{\qauthor}[1]{\par
{\raggedleft\upshape #1\qquad\hbox{}\par}\bigskip}

% JEM: All of this is so that headers in these sections aren't uppercase
\let\@ldtoc\tableofcontents
\renewcommand*\tableofcontents{{%
\renewcommand*\MakeUppercase[1]{##1}\@ldtoc}}
\let\@ldlof\listoffigures
\renewcommand*\listoffigures{{%
\renewcommand*\MakeUppercase[1]{##1}\@ldlof}}
\let\@ldlot\listoftables
\renewcommand*\listoftables{{%
\renewcommand*\MakeUppercase[1]{##1}\@ldlot}}
\let\@ldbib\bibliography
\renewcommand*\bibliography[1]{{%
\renewcommand*\MakeUppercase[1]{##1}\@ldbib{#1}}}

% JEM: for crazy multi character set quotes and Unicode and all
\usepackage{csquotes}
\usepackage{polyglossia}
\setdefaultlanguage{english}
\setotherlanguages{latin}
% e.g, for simplified Chinese:
% \setotherlanguages{chinese} % extend list above
% \newfontfamily\chinesefont[Script=CJK]{Noto Serif CJK SC}
% usage: \otherlanguage{chinese}北京市\end{otherlanguage}
%        \begin{chinese}北京市\end{chinese}
%        \textchinese{北京市}

% JEM: for highlighting corrections
\newlength{\mccurrentbaselineskip}
\newlength{\mccurrentparskip}
\newlength{\mccurrentparindent}
\newif\ifcorrections
\newif\ifm@csection
\m@csectionfalse
\correctionsfalse

\colorlet{shadecolor}{blue!20}

\usepackage{framed}
\OuterFrameSep=-9pt
\newenvironment{mccorrection}
{\ifcorrections\if@nobreak\m@csectiontrue\fi\begin{shaded}\ifm@csection\noindent\ignorespaces\fi\fi}
{\ifcorrections\end{shaded}\m@csectionfalse\ifx\@captype\@undefined\@nobreakfalse\fi\fi}

\usepackage{soul}
\sethlcolor{shadecolor}
\newcommand{\mccorrect}[1]{\ifcorrections\hl{#1}\else#1\fi}


% JEM: adjust start-of-chapter spacing
\renewcommand{\chapterheadstartvskip}{\vspace*{40pt}}
\renewcommand{\chapterheadendvskip}{\vspace{30pt}}


% JEM: combine all of the commands you run before appendices start
\newcommand{\startappendices}
  {\appendix\adjustmtc\noappendicestocpagenum\addappheadtotoc\appendixpage}


%
% Acronyms
%

\usepackage{acro}
\usepackage{pdfcomment}

\acsetup{
  pdfcomments/use = true,
  make-links = true,
  list/template = tabular
}

\newcommand{\DeclareAcronymWithTooltip}[3]{%
  \DeclareAcronym{#1}{%
    short = #2,
    long  = #3,
    pdfcomment = #3
  }%
}

%
% Colors (used on unibw.de)
%

\definecolor{unibwgray}{HTML}{282828}
\definecolor{unibworange}{HTML}{ED6600}
\definecolor{unibwblue}{HTML}{003366}
% ligther versions
\colorlet{unibwgraylight}{unibwgray!70}
\colorlet{unibworangelight}{unibworange!70}
\colorlet{unibwbluelight}{unibwblue!70}

%
% Boxed environments
%

\usepackage[most]{tcolorbox}
\usepackage{xparse}
\usepackage[titles]{tocloft}
\usepackage{etoolbox}
\usepackage[capitalize,noabbrev,nameinlink]{cleveref}

% Base style for all boxed environments
\tcbset{
  baseboxstyle/.style={
    enhanced,
    colframe=gray!50,
    colback=gray!5,
    boxrule=0.4pt,
    arc=1mm,
    % fontupper=\small,
    left=2mm, right=2mm, top=1mm, bottom=1mm,
    before skip=1.5em, after skip=1.5em,
    breakable,
    width=\linewidth,
    fonttitle=\scshape,
    coltitle=white,
    fontupper=\setlength{\baselineskip}{\textbaselineskip}
  }
}

% Macro: \newboxenv{id}{DisplayName}{DisplayNamePlural}{Color} - e.g., \newboxenv{excursus}{Excursus}{Excursuses}{gray}
\makeatletter
\NewDocumentCommand{\newboxenv}{mmmm}{
  \newcounter{#1}
  \expandafter\def\csname listname@#1\endcsname{List of #3}
  \expandafter\newlistof\expandafter{list#1}{#1}{\csname listname@#1\endcsname}

  \NewDocumentEnvironment{#1}{O{}}
   {
    \refstepcounter{#1}
    \def\boxenvlabel{#2~\arabic{#1}}
    \def\boxenvtitle{\boxenvlabel\IfValueT{##1}{:~##1}}
    \IfValueTF{##1}
      {\addcontentsline{#1}{list#1}{\protect\numberline{\arabic{#1}}{##1}}}
      {\addcontentsline{#1}{list#1}{\protect\numberline{\arabic{#1}}{\boxenvlabel}}}
    \tcolorbox[
      baseboxstyle,
      colframe=#4,
      colback=#4!5,
      title=\boxenvtitle,
    ]
   }
   {
    \endtcolorbox
   }

  \crefname{#1}{#2}{#3}
  \Crefname{#1}{#2}{#3}
}
\makeatother

% Macro: \listof{id} - e.g., \listof{excursus}
\NewDocumentCommand{\listof}{m}{
  \cleardoublepage
  \phantomsection
  \addcontentsline{toc}{chapter}{\csname listname@#1\endcsname}
  \csname listoflist#1\endcsname
}
