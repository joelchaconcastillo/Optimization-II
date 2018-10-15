all:
	pdflatex bare_jrnl.tex
	bibtex bare_jrnl.aux
	pdflatex bare_jrnl.tex
	pdflatex bare_jrnl.tex
	mv bare_jrnl.pdf Principal.pdf
#	latex boockchapter.tex
#	bibtex boockchapter
#	latex boockchapter.tex
#	latex boockchapter.tex
#	dvi2ps -o boockchapter.ps boockchapter.dvi
#	ps2pdf -sPAPERSIZE=letter -DMaxSubsetPct=100 -dCompatibilityLevel=1.4 -dPDFSETTINGS=/prepress -dSubsetFonts=true -dEmbedAllFonts=true boockchapter.ps CameraReady.pdf
	##pdflatex boockchapter.ps

clean:
	rm -f *.log *.dvi *.aux *.bbl *.blg *~
	
cleanall:
	rm -f *.log *.dvi *.aux *.bbl *.blg *~ Tesis_Multiobjetivo.pdf *.ps *.out
