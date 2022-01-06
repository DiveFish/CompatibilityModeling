VIRTUALENV_DIR=${PWD}/env
PIP=${VIRTUALENV_DIR}/bin/pip

all: virtualenv
	${PIP} install -r requirements.txt
	${PIP} install src/


virtualenv:
	if [ ! -e ${PIP} ]; then \
	python3 -m venv $(VIRTUALENV_DIR) --system-site-packages; \
	fi
	$(PIP) install --upgrade pip

clean:
	-rm -fv .DS_Store .coverage
	find . -name '*.pyc' -exec rm -fv {} \;
	find . -name '*.pyo' -exec rm -fv {} \;
	find . -depth -name '__pycache__' -exec rm -rfv {} \;

dist-clean: clean
	-rm -rfv ${VIRTUALENV_DIR} && \
	find . -depth -name '*.egg-info' -exec rm -rfv {} \;
