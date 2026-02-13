%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Masters/Doctoral Thesis
% LaTeX Class
% Version 1.6 (adapted for UniBw)
%
% Based on MastersDoctoralThesis by Vel (vel@latextemplates.com)
% License: CC BY-NC-SA 3.0
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{MastersDoctoralThesis}[2024/01/01 v1.6 Masters/Doctoral Thesis Class]

\newcommand{\classname}{MastersDoctoralThesis}

%----------------------------------------------------------------------------------------
% CLASS OPTIONS
%----------------------------------------------------------------------------------------

\providecommand{\baseclass}{book}

\newif\ifmc@nobind
\mc@nobindfalse
\DeclareOption{nobind}{\mc@nobindtrue}
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{\baseclass}}
\ProcessOptions\relax

\LoadClass[11pt,a4paper,twoside,openany]{\baseclass}

%----------------------------------------------------------------------------------------
% REQUIRED PACKAGES
%----------------------------------------------------------------------------------------

% For LuaLaTeX/XeLaTeX compatibility - fonts loaded in defines.tex
\RequirePackage{iftex}
\ifluatex
  % Using LuaLaTeX - fontspec will be loaded in defines.tex
\else\ifxetex
  % Using XeLaTeX - fontspec will be loaded in defines.tex
\else
  % PDFLaTeX fallback
  \RequirePackage[utf8]{inputenc}
  \RequirePackage[T1]{fontenc}
  \RequirePackage{mathpazo}
\fi\fi

\linespread{1.05}

\RequirePackage{graphicx}
\RequirePackage[english]{babel}
\RequirePackage{setspace}
\RequirePackage{longtable}
\RequirePackage{booktabs}
\RequirePackage[centerlast,small]{caption}
\RequirePackage{xcolor}
\RequirePackage{etoolbox}

%----------------------------------------------------------------------------------------
% MARGINS
%----------------------------------------------------------------------------------------

\RequirePackage{geometry}
\geometry{
  paper=a4paper,
  inner=3.5cm,
  outer=2.5cm,
  bindingoffset=0cm,
  top=2.5cm,
  bottom=2.5cm,
  headheight=14pt,
}

\raggedbottom

%----------------------------------------------------------------------------------------
% PENALTIES
%----------------------------------------------------------------------------------------

\doublehyphendemerits=10000
\brokenpenalty=10000
\widowpenalty=9999
\clubpenalty=9999
\interfootnotelinepenalty=9999

%----------------------------------------------------------------------------------------
% HEADERS AND FOOTERS
%----------------------------------------------------------------------------------------

\RequirePackage{fancyhdr}
\pagestyle{fancy}
\fancyhf{}

\renewcommand{\chaptermark}[1]{\markboth{\MakeUppercase{\thechapter.\ #1}}{}}
\renewcommand{\sectionmark}[1]{\markright{\MakeUppercase{\thesection.\ #1}}}

\fancyhead[LE,RO]{\thepage}
\fancyhead[RE]{\leftmark}
\fancyhead[LO]{\rightmark}

\fancypagestyle{plain}{%
  \fancyhf{}
  \fancyfoot[C]{\thepage}
  \renewcommand{\headrulewidth}{0pt}
}

%----------------------------------------------------------------------------------------
% THESIS INFORMATION COMMANDS
%----------------------------------------------------------------------------------------

\newcommand{\thesistitle}[1]{\def\@thesistitle{#1}\title{#1}}
\newcommand{\supervisor}[1]{\def\@supervisor{#1}}
\newcommand{\cosupervisor}[1]{\def\@cosupervisor{#1}}
\newcommand{\examiner}[1]{\def\@examiner{#1}}
\newcommand{\degree}[1]{\def\@degree{#1}}
\newcommand{\addresses}[1]{\def\@addresses{#1}}
\newcommand{\university}[1]{\def\@university{#1}}
\newcommand{\department}[1]{\def\@department{#1}}
\newcommand{\faculty}[1]{\def\@faculty{#1}}
\newcommand{\subject}[1]{\def\@subject{#1}}
\newcommand{\keywords}[1]{\def\@keywords{#1}}
\newcommand{\crest}[1]{\def\@crest{#1}}

% Defaults
\thesistitle{}
\supervisor{}
\cosupervisor{}
\examiner{}
\degree{}
\addresses{}
\university{}
\department{}
\faculty{}
\subject{}
\keywords{}
\crest{}

%----------------------------------------------------------------------------------------
% CHAPTER HEADINGS - COMPACT STYLE
%----------------------------------------------------------------------------------------

\def\@makechapterhead#1{%
  \vspace*{10\p@}%
  {\parindent \z@ \raggedright \normalfont
    \ifnum \c@secnumdepth >\m@ne
      \huge\bfseries \@chapapp\space \thechapter
      \par\nobreak
      \vskip 10\p@
    \fi
    \interlinepenalty\@M
    \Huge \bfseries #1\par\nobreak
    \vskip 20\p@
  }}

\def\@makeschapterhead#1{%
  \vspace*{10\p@}%
  {\parindent \z@ \raggedright
    \normalfont
    \interlinepenalty\@M
    \Huge \bfseries #1\par\nobreak
    \vskip 20\p@
  }}

%----------------------------------------------------------------------------------------
% PART HEADINGS WITH INTEGRATED INTRO
%----------------------------------------------------------------------------------------

% Storage for part introduction text
\newif\if@haspartintro
\@haspartintrofalse
\newcommand{\@partintro}{}
\newcommand{\partintro}[1]{%
  \renewcommand{\@partintro}{#1}%
  \@haspartintrotrue
}

% Redefine \@endpart - just go to next page (no blank verso page)
\def\@endpart{\vfil\newpage}

\def\@part[#1]#2{%
  \ifnum \c@secnumdepth >-2\relax
    \refstepcounter{part}%
    \addcontentsline{toc}{part}{\thepart\hspace{1em}#1}%
  \else
    \addcontentsline{toc}{part}{#1}%
  \fi
  \markboth{}{}%
  {\centering
   \interlinepenalty \@M
   \normalfont
   \ifnum \c@secnumdepth >-2\relax
     \huge\bfseries \partname\nobreakspace\thepart
     \par
     \vskip 20\p@
   \fi
   \Huge \bfseries #2\par}%
  % Include part intro if defined
  \if@haspartintro
    \vskip 40\p@
    \@partintro
    \global\@haspartintrofalse
    \global\renewcommand{\@partintro}{}%
  \fi
  \@endpart}

%----------------------------------------------------------------------------------------
% TITLE PAGE
%----------------------------------------------------------------------------------------

\renewcommand{\maketitle}{%
\begin{titlepage}
\begin{center}

\ifx\@crest\empty\else
  \@crest\\[1cm]
\fi

{\scshape\LARGE \@university \par}
\vspace{0.5cm}
{\scshape\large \@faculty \par}
{\scshape\large \@department \par}
\vspace{1.5cm}

{\huge\bfseries \@thesistitle \par}
\vspace{1cm}

{\Large \@degree \par}
\vspace{2cm}

{\large\itshape \@author \par}
\vfill

\ifx\@supervisor\empty\else
  {\large Supervised by: \@supervisor \par}
  \vspace{0.5cm}
\fi
\ifx\@examiner\empty\else
  {\large Examined by: \@examiner \par}
\fi

\vfill
{\large \today}

\end{center}
\end{titlepage}
}

%----------------------------------------------------------------------------------------
% ABSTRACT
%----------------------------------------------------------------------------------------

\newenvironment{abstract}{%
  \cleardoublepage
  \thispagestyle{empty}
  \begin{center}
    \ifx\@crest\empty\else
      \@crest\\[0.5cm]
    \fi
    {\normalsize \MakeUppercase{\@university} \par}
    {\normalsize \@faculty \par}
    {\normalsize \@department \par}
    \vspace{0.5cm}
    {\normalsize \@degree \par}
    \vspace{1cm}
    {\normalsize\bfseries \@thesistitle \par}
    \vspace{0.3cm}
    {\normalsize by \@author \par}
    \vspace{1cm}
    {\huge\textit{Abstract} \par}
  \end{center}
  \vspace{1cm}
}{%
  \vfill\null
  \cleardoublepage
}

% German abstract (Kurzfassung)
\newenvironment{kurzfassung}{%
  \cleardoublepage
  \thispagestyle{empty}
  \begin{center}
    \ifx\@crest\empty\else
      \@crest\\[0.5cm]
    \fi
    {\normalsize \MakeUppercase{\@university} \par}
    {\normalsize \@faculty \par}
    {\normalsize \@department \par}
    \vspace{0.5cm}
    {\normalsize \@degree \par}
    \vspace{1cm}
    {\normalsize\bfseries \@thesistitle \par}
    \vspace{0.3cm}
    {\normalsize von \@author \par}
    \vspace{1cm}
    {\huge\textit{Kurzfassung} \par}
  \end{center}
  \vspace{1cm}
}{%
  \vfill\null
  \cleardoublepage
}

%----------------------------------------------------------------------------------------
% ACKNOWLEDGEMENTS
%----------------------------------------------------------------------------------------

\newenvironment{acknowledgements}{%
  \cleardoublepage
  \thispagestyle{empty}
  \begin{center}
    {\huge\textit{Acknowledgements} \par}
  \end{center}
  \vspace{1cm}
}{%
  \vfill\null
  \cleardoublepage
}

%----------------------------------------------------------------------------------------
% DECLARATION
%----------------------------------------------------------------------------------------

\newenvironment{declaration}{%
  \cleardoublepage
  \thispagestyle{empty}
  \begin{center}
    {\huge\bfseries Declaration of Authorship \par}
  \end{center}
  \vspace{1cm}
}{%
  \vfill\null
  \cleardoublepage
}

%----------------------------------------------------------------------------------------
% DEDICATION
%----------------------------------------------------------------------------------------

\newenvironment{dedication}{%
  \cleardoublepage
  \thispagestyle{empty}
  \vspace*{\fill}
  \begin{center}
  \itshape
}{%
  \end{center}
  \vspace*{\fill}
  \cleardoublepage
}

%----------------------------------------------------------------------------------------
% ABBREVIATIONS
%----------------------------------------------------------------------------------------

\newenvironment{abbreviations}{%
  \cleardoublepage
  \chapter*{Abbreviations}
  \addcontentsline{toc}{chapter}{Abbreviations}
  \begin{longtable}[l]{@{}ll@{}}
}{%
  \end{longtable}
}

% Hyperref should be loaded in the main document or defines.tex
% to avoid option clashes

\endinput
