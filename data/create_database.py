# To create your database
import numpy as np

# Simulate dummy entries
database = {
    "temp": np.random.rand(64*128),
    "Shubham": np.random.rand(64*128),
    "Lobhesh": np.random.rand(64*128)
}

np.save("data/database.npy", database)
