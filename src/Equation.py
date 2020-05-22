import numpy as np
from classify import classify_getDigit, classify_getSymbol
import matplotlib.pyplot as plt


class Equation(object):
	"""Keeps track of robot position an triggers symbol and digit classification when needd"""
	# listObj is of form :
	# listObj[i]["pos"] = (xi, yi)
	# listObj[i]["img"] = np.array() # 28x28 pixel binary


	def __init__(self, listObj, args):
		self.prevDigit = False
		self.inside = False
		self.currentObj = listObj[0]
		self.equ = ""

		self.listObj = listObj
		self.areaInRad = 30 # max distance to symbol center to be considered over it
		self.areaOutRad = self.areaInRad + 10 # min distance away from symbol to be conisderd outside (histeresis)
		self.args = args


	def newRobPos(self, pos):
		if self.inside :
			d = np.linalg.norm(np.subtract(pos, self.currentObj["pos"]))
			if d > self.areaOutRad :
				self.inside = False

		else :
			mind = 10000000
			for obj in self.listObj :
				d = np.linalg.norm(np.subtract(pos, obj["pos"]))
				if d < mind and d < self.areaInRad :
					self.inside = True
					self.currentObj = obj
					mind = d

			if self.inside : #got inside
				if self.prevDigit :
					symb = classify_getSymbol(self.currentObj["img"])
					# fig, ax = plt.subplots()
					# ax.imshow(self.currentObj["img"])
					# ax.set_title("Lbl : {}".format(symb))
					# plt.show()
					res = ''
					if symb == '=' : res = str(np.round(eval(self.equ),2))
					self.equ = ''.join([self.equ, symb, res])
					self.prevDigit = False
				else :
					self.equ = ''.join([self.equ, classify_getDigit(self.currentObj["img"])])
					# fig, ax = plt.subplots()
					# ax.imshow(self.currentObj["img"])
					# ax.set_title("Lbl : {}".format(classify_getDigit(self.currentObj["img"], self.currentObj["label"])))
					# plt.show()
					self.prevDigit = True

		return self.equ

			
		