x = data["Area (In Sq. feet)"]
y = data["Prices"]

plt.scatter(x,y)
plt.xlabel("Area")
plt.ylabel("Prices")
plt.title("House Prices")
plt.show()