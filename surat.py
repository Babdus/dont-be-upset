from PIL import Image

class Surat:

	def __init__(self, path, m, p):
		if path is None:
			self.image = Image.new("RGB", (m*p, m*p), "white")
		else:
			self.image = Image.open(path)
		if self.image.size[0] != 512 or self.image.size[1] != 512:
			raise ValueError(f'The resolution of the image on path {path} is not 512×512!')
		self.l = self.image.size[0]
		self.m = m if m is not None else self.l / p
		self.p = p if p is not None else self.l / m

	def get_fragment(self, i, j, n=None):
		if n is not None and (i is None or i is None):
			i = int(n / self.m)
			j = int(n % self.m)
		return self.image.crop((j*self.p, i*self.p, (j+1)*self.p, (i+1)*self.p))

	def paste_fragment(self, fragment, i, j, n=None):
		if n is not None and (i is None or i is None):
			i = int(n / self.m)
			j = int(n % self.m)
		return self.image.paste(fragment, (j*self.p, i*self.p, (j+1)*self.p, (i+1)*self.p))

	def rearrange(self, sequence, reverse=False):
		new_surat = Surat(None, self.m, self.p)
		for i, n in enumerate(sequence):
			num = i if reverse else int(n)
			idx = int(n) if reverse else i
			fragment = self.get_fragment(None, None, num)
			new_surat.paste_fragment(fragment, None, None, idx)
		return new_surat

	def save(self, path):
		self.image.save(path, 'PNG')

	def get_pair(self, n1, n2, left=True, transpose=True):
		frag1 = self.get_fragment(None, None, n1)
		frag2 = self.get_fragment(None, None, n2)
		p = self.p
		if left:
			pair = Image.new("RGB", (p*2, p), "white")
			pair.paste(frag1, (0, 0, p, p))
			pair.paste(frag2, (p, 0, p*2, p))
		else:
			pair = Image.new("RGB", (p, p*2), "white")
			pair.paste(frag1, (0, 0, p, p))
			pair.paste(frag2, (0, p, p, p*2))
			if transpose:
				pair = pair.transpose(Image.ROTATE_90)
		return pair