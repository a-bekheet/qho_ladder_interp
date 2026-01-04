from qho_ladder_interp.tokens import VOCAB, encode, decode

print(len(VOCAB))
print(VOCAB[:10])

x = ["<BOS>", "OP=LOWER", "S17", "<SEP>", "S16", "<EOS>"]
ids = encode(x)
x_roundtrip = decode(ids)

print(x)
print(x_roundtrip)
