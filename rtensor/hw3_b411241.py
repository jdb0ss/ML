# 1.
def make_album(artist, title):
    return({'artist': artist, 'title': title})

print(make_album("BTS", "Map of the Soul: 7"))

# 2.

def make_album2(artist, title, num_tracks=''):
    if num_tracks == '':
        return({'artist': artist, 'title': title})
    else:
        return({'artist': artist, 'title': title, 'num_tracks': num_tracks})


print(make_album2("BTS", "Map of the Soul: 7"))
print(make_album2("BTS", "Map of the Soul: 7", "32"))

# 3.

def build_profile(first, last, **user_info):
    return({'first': first, 'last': last, 'address': user_info['address'], 'phone': user_info['phone']})


print(build_profile('홍', '성민', address='여의도', phone='01077774444'))

# 4.
class Restaurant():
    def __init__(self, name, cuisine):
        self.name = name
        self.cuisine = cuisine
        self.number_served = 0

    def set_number_served(self, k):
        self.number_served = k

    def get_number_served(self):
        return self.number_served

    def increment_number_served(self, inc):
        self.number_served += inc


bon = Restaurant("Bon Giorno", "Italian")
bon.number_served = 4
print(bon.get_number_served())
bon.set_number_served(8)
print(bon.get_number_served())
bon.increment_number_served(6)
print(bon.get_number_served())

# 5.

class IceCreamStand:
    def __init__(self, *flavors):
        super().__init__()
        self.flavors = ['vanilla', 'strawberry', 'chocolate']

    def show_flavors(self):
        print(self.flavors)

icec = IceCreamStand("Bon", "Ital")
icec.show_flavors()
