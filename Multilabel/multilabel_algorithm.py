# https://pystruct.github.io/generated/pystruct.models.MultiLabelClf.html#pystruct.models.MultiLabelClf
from pystruct.datasets import load_scene
from pystruct.learners import NSlackSSVM
from pystruct.models import MultiLabelClf

data = load_scene()
X_train, X_test = data['X_train'], data['X_test']
y_train, y_test = data['y_train'], data['y_test']


clf = NSlackSSVM(MultiLabelClf())

clf.fit(X_train, y_train)
clf.predict(X_test)
print(clf.score(X_test, y_test))