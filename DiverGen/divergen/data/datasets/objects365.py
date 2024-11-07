# Copyright (c) Facebook, Inc. and its affiliates.
from detectron2.data.datasets.register_coco import register_coco_instances
import os

# categories_v2 = [
#     {'id': 1, 'name': 'Person'},
#     {'id': 2, 'name': 'Sneakers'},
#     {'id': 3, 'name': 'Chair'},
#     {'id': 4, 'name': 'Other Shoes'},
#     {'id': 5, 'name': 'Hat'},
#     {'id': 6, 'name': 'Car'},
#     {'id': 7, 'name': 'Lamp'},
#     {'id': 8, 'name': 'Glasses'},
#     {'id': 9, 'name': 'Bottle'},
#     {'id': 10, 'name': 'Desk'},
#     {'id': 11, 'name': 'Cup'},
#     {'id': 12, 'name': 'Street Lights'},
#     {'id': 13, 'name': 'Cabinet/shelf'},
#     {'id': 14, 'name': 'Handbag/Satchel'},
#     {'id': 15, 'name': 'Bracelet'},
#     {'id': 16, 'name': 'Plate'},
#     {'id': 17, 'name': 'Picture/Frame'},
#     {'id': 18, 'name': 'Helmet'},
#     {'id': 19, 'name': 'Book'},
#     {'id': 20, 'name': 'Gloves'},
#     {'id': 21, 'name': 'Storage box'},
#     {'id': 22, 'name': 'Boat'},
#     {'id': 23, 'name': 'Leather Shoes'},
#     {'id': 24, 'name': 'Flower'},
#     {'id': 25, 'name': 'Bench'},
#     {'id': 26, 'name': 'Potted Plant'},
#     {'id': 27, 'name': 'Bowl/Basin'},
#     {'id': 28, 'name': 'Flag'},
#     {'id': 29, 'name': 'Pillow'},
#     {'id': 30, 'name': 'Boots'},
#     {'id': 31, 'name': 'Vase'},
#     {'id': 32, 'name': 'Microphone'},
#     {'id': 33, 'name': 'Necklace'},
#     {'id': 34, 'name': 'Ring'},
#     {'id': 35, 'name': 'SUV'},
#     {'id': 36, 'name': 'Wine Glass'},
#     {'id': 37, 'name': 'Belt'},
#     {'id': 38, 'name': 'Moniter/TV'},
#     {'id': 39, 'name': 'Backpack'},
#     {'id': 40, 'name': 'Umbrella'},
#     {'id': 41, 'name': 'Traffic Light'},
#     {'id': 42, 'name': 'Speaker'},
#     {'id': 43, 'name': 'Watch'},
#     {'id': 44, 'name': 'Tie'},
#     {'id': 45, 'name': 'Trash bin Can'},
#     {'id': 46, 'name': 'Slippers'},
#     {'id': 47, 'name': 'Bicycle'},
#     {'id': 48, 'name': 'Stool'},
#     {'id': 49, 'name': 'Barrel/bucket'},
#     {'id': 50, 'name': 'Van'},
#     {'id': 51, 'name': 'Couch'},
#     {'id': 52, 'name': 'Sandals'},
#     {'id': 53, 'name': 'Bakset'},
#     {'id': 54, 'name': 'Drum'},
#     {'id': 55, 'name': 'Pen/Pencil'},
#     {'id': 56, 'name': 'Bus'},
#     {'id': 57, 'name': 'Wild Bird'},
#     {'id': 58, 'name': 'High Heels'},
#     {'id': 59, 'name': 'Motorcycle'},
#     {'id': 60, 'name': 'Guitar'},
#     {'id': 61, 'name': 'Carpet'},
#     {'id': 62, 'name': 'Cell Phone'},
#     {'id': 63, 'name': 'Bread'},
#     {'id': 64, 'name': 'Camera'},
#     {'id': 65, 'name': 'Canned'},
#     {'id': 66, 'name': 'Truck'},
#     {'id': 67, 'name': 'Traffic cone'},
#     {'id': 68, 'name': 'Cymbal'},
#     {'id': 69, 'name': 'Lifesaver'},
#     {'id': 70, 'name': 'Towel'},
#     {'id': 71, 'name': 'Stuffed Toy'},
#     {'id': 72, 'name': 'Candle'},
#     {'id': 73, 'name': 'Sailboat'},
#     {'id': 74, 'name': 'Laptop'},
#     {'id': 75, 'name': 'Awning'},
#     {'id': 76, 'name': 'Bed'},
#     {'id': 77, 'name': 'Faucet'},
#     {'id': 78, 'name': 'Tent'},
#     {'id': 79, 'name': 'Horse'},
#     {'id': 80, 'name': 'Mirror'},
#     {'id': 81, 'name': 'Power outlet'},
#     {'id': 82, 'name': 'Sink'},
#     {'id': 83, 'name': 'Apple'},
#     {'id': 84, 'name': 'Air Conditioner'},
#     {'id': 85, 'name': 'Knife'},
#     {'id': 86, 'name': 'Hockey Stick'},
#     {'id': 87, 'name': 'Paddle'},
#     {'id': 88, 'name': 'Pickup Truck'},
#     {'id': 89, 'name': 'Fork'},
#     {'id': 90, 'name': 'Traffic Sign'},
#     {'id': 91, 'name': 'Ballon'},
#     {'id': 92, 'name': 'Tripod'},
#     {'id': 93, 'name': 'Dog'},
#     {'id': 94, 'name': 'Spoon'},
#     {'id': 95, 'name': 'Clock'},
#     {'id': 96, 'name': 'Pot'},
#     {'id': 97, 'name': 'Cow'},
#     {'id': 98, 'name': 'Cake'},
#     {'id': 99, 'name': 'Dinning Table'},
#     {'id': 100, 'name': 'Sheep'},
#     {'id': 101, 'name': 'Hanger'},
#     {'id': 102, 'name': 'Blackboard/Whiteboard'},
#     {'id': 103, 'name': 'Napkin'},
#     {'id': 104, 'name': 'Other Fish'},
#     {'id': 105, 'name': 'Orange/Tangerine'},
#     {'id': 106, 'name': 'Toiletry'},
#     {'id': 107, 'name': 'Keyboard'},
#     {'id': 108, 'name': 'Tomato'},
#     {'id': 109, 'name': 'Lantern'},
#     {'id': 110, 'name': 'Machinery Vehicle'},
#     {'id': 111, 'name': 'Fan'},
#     {'id': 112, 'name': 'Green Vegetables'},
#     {'id': 113, 'name': 'Banana'},
#     {'id': 114, 'name': 'Baseball Glove'},
#     {'id': 115, 'name': 'Airplane'},
#     {'id': 116, 'name': 'Mouse'},
#     {'id': 117, 'name': 'Train'},
#     {'id': 118, 'name': 'Pumpkin'},
#     {'id': 119, 'name': 'Soccer'},
#     {'id': 120, 'name': 'Skiboard'},
#     {'id': 121, 'name': 'Luggage'},
#     {'id': 122, 'name': 'Nightstand'},
#     {'id': 123, 'name': 'Tea pot'},
#     {'id': 124, 'name': 'Telephone'},
#     {'id': 125, 'name': 'Trolley'},
#     {'id': 126, 'name': 'Head Phone'},
#     {'id': 127, 'name': 'Sports Car'},
#     {'id': 128, 'name': 'Stop Sign'},
#     {'id': 129, 'name': 'Dessert'},
#     {'id': 130, 'name': 'Scooter'},
#     {'id': 131, 'name': 'Stroller'},
#     {'id': 132, 'name': 'Crane'},
#     {'id': 133, 'name': 'Remote'},
#     {'id': 134, 'name': 'Refrigerator'},
#     {'id': 135, 'name': 'Oven'},
#     {'id': 136, 'name': 'Lemon'},
#     {'id': 137, 'name': 'Duck'},
#     {'id': 138, 'name': 'Baseball Bat'},
#     {'id': 139, 'name': 'Surveillance Camera'},
#     {'id': 140, 'name': 'Cat'},
#     {'id': 141, 'name': 'Jug'},
#     {'id': 142, 'name': 'Broccoli'},
#     {'id': 143, 'name': 'Piano'},
#     {'id': 144, 'name': 'Pizza'},
#     {'id': 145, 'name': 'Elephant'},
#     {'id': 146, 'name': 'Skateboard'},
#     {'id': 147, 'name': 'Surfboard'},
#     {'id': 148, 'name': 'Gun'},
#     {'id': 149, 'name': 'Skating and Skiing shoes'},
#     {'id': 150, 'name': 'Gas stove'},
#     {'id': 151, 'name': 'Donut'},
#     {'id': 152, 'name': 'Bow Tie'},
#     {'id': 153, 'name': 'Carrot'},
#     {'id': 154, 'name': 'Toilet'},
#     {'id': 155, 'name': 'Kite'},
#     {'id': 156, 'name': 'Strawberry'},
#     {'id': 157, 'name': 'Other Balls'},
#     {'id': 158, 'name': 'Shovel'},
#     {'id': 159, 'name': 'Pepper'},
#     {'id': 160, 'name': 'Computer Box'},
#     {'id': 161, 'name': 'Toilet Paper'},
#     {'id': 162, 'name': 'Cleaning Products'},
#     {'id': 163, 'name': 'Chopsticks'},
#     {'id': 164, 'name': 'Microwave'},
#     {'id': 165, 'name': 'Pigeon'},
#     {'id': 166, 'name': 'Baseball'},
#     {'id': 167, 'name': 'Cutting/chopping Board'},
#     {'id': 168, 'name': 'Coffee Table'},
#     {'id': 169, 'name': 'Side Table'},
#     {'id': 170, 'name': 'Scissors'},
#     {'id': 171, 'name': 'Marker'},
#     {'id': 172, 'name': 'Pie'},
#     {'id': 173, 'name': 'Ladder'},
#     {'id': 174, 'name': 'Snowboard'},
#     {'id': 175, 'name': 'Cookies'},
#     {'id': 176, 'name': 'Radiator'},
#     {'id': 177, 'name': 'Fire Hydrant'},
#     {'id': 178, 'name': 'Basketball'},
#     {'id': 179, 'name': 'Zebra'},
#     {'id': 180, 'name': 'Grape'},
#     {'id': 181, 'name': 'Giraffe'},
#     {'id': 182, 'name': 'Potato'},
#     {'id': 183, 'name': 'Sausage'},
#     {'id': 184, 'name': 'Tricycle'},
#     {'id': 185, 'name': 'Violin'},
#     {'id': 186, 'name': 'Egg'},
#     {'id': 187, 'name': 'Fire Extinguisher'},
#     {'id': 188, 'name': 'Candy'},
#     {'id': 189, 'name': 'Fire Truck'},
#     {'id': 190, 'name': 'Billards'},
#     {'id': 191, 'name': 'Converter'},
#     {'id': 192, 'name': 'Bathtub'},
#     {'id': 193, 'name': 'Wheelchair'},
#     {'id': 194, 'name': 'Golf Club'},
#     {'id': 195, 'name': 'Briefcase'},
#     {'id': 196, 'name': 'Cucumber'},
#     {'id': 197, 'name': 'Cigar/Cigarette '},
#     {'id': 198, 'name': 'Paint Brush'},
#     {'id': 199, 'name': 'Pear'},
#     {'id': 200, 'name': 'Heavy Truck'},
#     {'id': 201, 'name': 'Hamburger'},
#     {'id': 202, 'name': 'Extractor'},
#     {'id': 203, 'name': 'Extention Cord'},
#     {'id': 204, 'name': 'Tong'},
#     {'id': 205, 'name': 'Tennis Racket'},
#     {'id': 206, 'name': 'Folder'},
#     {'id': 207, 'name': 'American Football'},
#     {'id': 208, 'name': 'earphone'},
#     {'id': 209, 'name': 'Mask'},
#     {'id': 210, 'name': 'Kettle'},
#     {'id': 211, 'name': 'Tennis'},
#     {'id': 212, 'name': 'Ship'},
#     {'id': 213, 'name': 'Swing'},
#     {'id': 214, 'name': 'Coffee Machine'},
#     {'id': 215, 'name': 'Slide'},
#     {'id': 216, 'name': 'Carriage'},
#     {'id': 217, 'name': 'Onion'},
#     {'id': 218, 'name': 'Green beans'},
#     {'id': 219, 'name': 'Projector'},
#     {'id': 220, 'name': 'Frisbee'},
#     {'id': 221, 'name': 'Washing Machine/Drying Machine'},
#     {'id': 222, 'name': 'Chicken'},
#     {'id': 223, 'name': 'Printer'},
#     {'id': 224, 'name': 'Watermelon'},
#     {'id': 225, 'name': 'Saxophone'},
#     {'id': 226, 'name': 'Tissue'},
#     {'id': 227, 'name': 'Toothbrush'},
#     {'id': 228, 'name': 'Ice cream'},
#     {'id': 229, 'name': 'Hotair ballon'},
#     {'id': 230, 'name': 'Cello'},
#     {'id': 231, 'name': 'French Fries'},
#     {'id': 232, 'name': 'Scale'},
#     {'id': 233, 'name': 'Trophy'},
#     {'id': 234, 'name': 'Cabbage'},
#     {'id': 235, 'name': 'Hot dog'},
#     {'id': 236, 'name': 'Blender'},
#     {'id': 237, 'name': 'Peach'},
#     {'id': 238, 'name': 'Rice'},
#     {'id': 239, 'name': 'Wallet/Purse'},
#     {'id': 240, 'name': 'Volleyball'},
#     {'id': 241, 'name': 'Deer'},
#     {'id': 242, 'name': 'Goose'},
#     {'id': 243, 'name': 'Tape'},
#     {'id': 244, 'name': 'Tablet'},
#     {'id': 245, 'name': 'Cosmetics'},
#     {'id': 246, 'name': 'Trumpet'},
#     {'id': 247, 'name': 'Pineapple'},
#     {'id': 248, 'name': 'Golf Ball'},
#     {'id': 249, 'name': 'Ambulance'},
#     {'id': 250, 'name': 'Parking meter'},
#     {'id': 251, 'name': 'Mango'},
#     {'id': 252, 'name': 'Key'},
#     {'id': 253, 'name': 'Hurdle'},
#     {'id': 254, 'name': 'Fishing Rod'},
#     {'id': 255, 'name': 'Medal'},
#     {'id': 256, 'name': 'Flute'},
#     {'id': 257, 'name': 'Brush'},
#     {'id': 258, 'name': 'Penguin'},
#     {'id': 259, 'name': 'Megaphone'},
#     {'id': 260, 'name': 'Corn'},
#     {'id': 261, 'name': 'Lettuce'},
#     {'id': 262, 'name': 'Garlic'},
#     {'id': 263, 'name': 'Swan'},
#     {'id': 264, 'name': 'Helicopter'},
#     {'id': 265, 'name': 'Green Onion'},
#     {'id': 266, 'name': 'Sandwich'},
#     {'id': 267, 'name': 'Nuts'},
#     {'id': 268, 'name': 'Speed Limit Sign'},
#     {'id': 269, 'name': 'Induction Cooker'},
#     {'id': 270, 'name': 'Broom'},
#     {'id': 271, 'name': 'Trombone'},
#     {'id': 272, 'name': 'Plum'},
#     {'id': 273, 'name': 'Rickshaw'},
#     {'id': 274, 'name': 'Goldfish'},
#     {'id': 275, 'name': 'Kiwi fruit'},
#     {'id': 276, 'name': 'Router/modem'},
#     {'id': 277, 'name': 'Poker Card'},
#     {'id': 278, 'name': 'Toaster'},
#     {'id': 279, 'name': 'Shrimp'},
#     {'id': 280, 'name': 'Sushi'},
#     {'id': 281, 'name': 'Cheese'},
#     {'id': 282, 'name': 'Notepaper'},
#     {'id': 283, 'name': 'Cherry'},
#     {'id': 284, 'name': 'Pliers'},
#     {'id': 285, 'name': 'CD'},
#     {'id': 286, 'name': 'Pasta'},
#     {'id': 287, 'name': 'Hammer'},
#     {'id': 288, 'name': 'Cue'},
#     {'id': 289, 'name': 'Avocado'},
#     {'id': 290, 'name': 'Hamimelon'},
#     {'id': 291, 'name': 'Flask'},
#     {'id': 292, 'name': 'Mushroon'},
#     {'id': 293, 'name': 'Screwdriver'},
#     {'id': 294, 'name': 'Soap'},
#     {'id': 295, 'name': 'Recorder'},
#     {'id': 296, 'name': 'Bear'},
#     {'id': 297, 'name': 'Eggplant'},
#     {'id': 298, 'name': 'Board Eraser'},
#     {'id': 299, 'name': 'Coconut'},
#     {'id': 300, 'name': 'Tape Measur/ Ruler'},
#     {'id': 301, 'name': 'Pig'},
#     {'id': 302, 'name': 'Showerhead'},
#     {'id': 303, 'name': 'Globe'},
#     {'id': 304, 'name': 'Chips'},
#     {'id': 305, 'name': 'Steak'},
#     {'id': 306, 'name': 'Crosswalk Sign'},
#     {'id': 307, 'name': 'Stapler'},
#     {'id': 308, 'name': 'Campel'},
#     {'id': 309, 'name': 'Formula 1 '},
#     {'id': 310, 'name': 'Pomegranate'},
#     {'id': 311, 'name': 'Dishwasher'},
#     {'id': 312, 'name': 'Crab'},
#     {'id': 313, 'name': 'Hoverboard'},
#     {'id': 314, 'name': 'Meat ball'},
#     {'id': 315, 'name': 'Rice Cooker'},
#     {'id': 316, 'name': 'Tuba'},
#     {'id': 317, 'name': 'Calculator'},
#     {'id': 318, 'name': 'Papaya'},
#     {'id': 319, 'name': 'Antelope'},
#     {'id': 320, 'name': 'Parrot'},
#     {'id': 321, 'name': 'Seal'},
#     {'id': 322, 'name': 'Buttefly'},
#     {'id': 323, 'name': 'Dumbbell'},
#     {'id': 324, 'name': 'Donkey'},
#     {'id': 325, 'name': 'Lion'},
#     {'id': 326, 'name': 'Urinal'},
#     {'id': 327, 'name': 'Dolphin'},
#     {'id': 328, 'name': 'Electric Drill'},
#     {'id': 329, 'name': 'Hair Dryer'},
#     {'id': 330, 'name': 'Egg tart'},
#     {'id': 331, 'name': 'Jellyfish'},
#     {'id': 332, 'name': 'Treadmill'},
#     {'id': 333, 'name': 'Lighter'},
#     {'id': 334, 'name': 'Grapefruit'},
#     {'id': 335, 'name': 'Game board'},
#     {'id': 336, 'name': 'Mop'},
#     {'id': 337, 'name': 'Radish'},
#     {'id': 338, 'name': 'Baozi'},
#     {'id': 339, 'name': 'Target'},
#     {'id': 340, 'name': 'French'},
#     {'id': 341, 'name': 'Spring Rolls'},
#     {'id': 342, 'name': 'Monkey'},
#     {'id': 343, 'name': 'Rabbit'},
#     {'id': 344, 'name': 'Pencil Case'},
#     {'id': 345, 'name': 'Yak'},
#     {'id': 346, 'name': 'Red Cabbage'},
#     {'id': 347, 'name': 'Binoculars'},
#     {'id': 348, 'name': 'Asparagus'},
#     {'id': 349, 'name': 'Barbell'},
#     {'id': 350, 'name': 'Scallop'},
#     {'id': 351, 'name': 'Noddles'},
#     {'id': 352, 'name': 'Comb'},
#     {'id': 353, 'name': 'Dumpling'},
#     {'id': 354, 'name': 'Oyster'},
#     {'id': 355, 'name': 'Table Teniis paddle'},
#     {'id': 356, 'name': 'Cosmetics Brush/Eyeliner Pencil'},
#     {'id': 357, 'name': 'Chainsaw'},
#     {'id': 358, 'name': 'Eraser'},
#     {'id': 359, 'name': 'Lobster'},
#     {'id': 360, 'name': 'Durian'},
#     {'id': 361, 'name': 'Okra'},
#     {'id': 362, 'name': 'Lipstick'},
#     {'id': 363, 'name': 'Cosmetics Mirror'},
#     {'id': 364, 'name': 'Curling'},
#     {'id': 365, 'name': 'Table Tennis '},
# ]

'''
The official Objects365 category names contains typos.
Below is a manual fix.
'''
categories_v2_fix = [
  {'id': 1, 'name': 'Person'},
  {'id': 2, 'name': 'Sneakers'},
  {'id': 3, 'name': 'Chair'},
  {'id': 4, 'name': 'Other Shoes'},
  {'id': 5, 'name': 'Hat'},
  {'id': 6, 'name': 'Car'},
  {'id': 7, 'name': 'Lamp'},
  {'id': 8, 'name': 'Glasses'},
  {'id': 9, 'name': 'Bottle'},
  {'id': 10, 'name': 'Desk'},
  {'id': 11, 'name': 'Cup'},
  {'id': 12, 'name': 'Street Lights'},
  {'id': 13, 'name': 'Cabinet/shelf'},
  {'id': 14, 'name': 'Handbag/Satchel'},
  {'id': 15, 'name': 'Bracelet'},
  {'id': 16, 'name': 'Plate'},
  {'id': 17, 'name': 'Picture/Frame'},
  {'id': 18, 'name': 'Helmet'},
  {'id': 19, 'name': 'Book'},
  {'id': 20, 'name': 'Gloves'},
  {'id': 21, 'name': 'Storage box'},
  {'id': 22, 'name': 'Boat'},
  {'id': 23, 'name': 'Leather Shoes'},
  {'id': 24, 'name': 'Flower'},
  {'id': 25, 'name': 'Bench'},
  {'id': 26, 'name': 'Potted Plant'},
  {'id': 27, 'name': 'Bowl/Basin'},
  {'id': 28, 'name': 'Flag'},
  {'id': 29, 'name': 'Pillow'},
  {'id': 30, 'name': 'Boots'},
  {'id': 31, 'name': 'Vase'},
  {'id': 32, 'name': 'Microphone'},
  {'id': 33, 'name': 'Necklace'},
  {'id': 34, 'name': 'Ring'},
  {'id': 35, 'name': 'SUV'},
  {'id': 36, 'name': 'Wine Glass'},
  {'id': 37, 'name': 'Belt'},
  {'id': 38, 'name': 'Monitor/TV'},
  {'id': 39, 'name': 'Backpack'},
  {'id': 40, 'name': 'Umbrella'},
  {'id': 41, 'name': 'Traffic Light'},
  {'id': 42, 'name': 'Speaker'},
  {'id': 43, 'name': 'Watch'},
  {'id': 44, 'name': 'Tie'},
  {'id': 45, 'name': 'Trash bin Can'},
  {'id': 46, 'name': 'Slippers'},
  {'id': 47, 'name': 'Bicycle'},
  {'id': 48, 'name': 'Stool'},
  {'id': 49, 'name': 'Barrel/bucket'},
  {'id': 50, 'name': 'Van'},
  {'id': 51, 'name': 'Couch'},
  {'id': 52, 'name': 'Sandals'},
  {'id': 53, 'name': 'Basket'},
  {'id': 54, 'name': 'Drum'},
  {'id': 55, 'name': 'Pen/Pencil'},
  {'id': 56, 'name': 'Bus'},
  {'id': 57, 'name': 'Wild Bird'},
  {'id': 58, 'name': 'High Heels'},
  {'id': 59, 'name': 'Motorcycle'},
  {'id': 60, 'name': 'Guitar'},
  {'id': 61, 'name': 'Carpet'},
  {'id': 62, 'name': 'Cell Phone'},
  {'id': 63, 'name': 'Bread'},
  {'id': 64, 'name': 'Camera'},
  {'id': 65, 'name': 'Canned'},
  {'id': 66, 'name': 'Truck'},
  {'id': 67, 'name': 'Traffic cone'},
  {'id': 68, 'name': 'Cymbal'},
  {'id': 69, 'name': 'Lifesaver'},
  {'id': 70, 'name': 'Towel'},
  {'id': 71, 'name': 'Stuffed Toy'},
  {'id': 72, 'name': 'Candle'},
  {'id': 73, 'name': 'Sailboat'},
  {'id': 74, 'name': 'Laptop'},
  {'id': 75, 'name': 'Awning'},
  {'id': 76, 'name': 'Bed'},
  {'id': 77, 'name': 'Faucet'},
  {'id': 78, 'name': 'Tent'},
  {'id': 79, 'name': 'Horse'},
  {'id': 80, 'name': 'Mirror'},
  {'id': 81, 'name': 'Power outlet'},
  {'id': 82, 'name': 'Sink'},
  {'id': 83, 'name': 'Apple'},
  {'id': 84, 'name': 'Air Conditioner'},
  {'id': 85, 'name': 'Knife'},
  {'id': 86, 'name': 'Hockey Stick'},
  {'id': 87, 'name': 'Paddle'},
  {'id': 88, 'name': 'Pickup Truck'},
  {'id': 89, 'name': 'Fork'},
  {'id': 90, 'name': 'Traffic Sign'},
  {'id': 91, 'name': 'Ballon'},
  {'id': 92, 'name': 'Tripod'},
  {'id': 93, 'name': 'Dog'},
  {'id': 94, 'name': 'Spoon'},
  {'id': 95, 'name': 'Clock'},
  {'id': 96, 'name': 'Pot'},
  {'id': 97, 'name': 'Cow'},
  {'id': 98, 'name': 'Cake'},
  {'id': 99, 'name': 'Dining Table'},
  {'id': 100, 'name': 'Sheep'},
  {'id': 101, 'name': 'Hanger'},
  {'id': 102, 'name': 'Blackboard/Whiteboard'},
  {'id': 103, 'name': 'Napkin'},
  {'id': 104, 'name': 'Other Fish'},
  {'id': 105, 'name': 'Orange/Tangerine'},
  {'id': 106, 'name': 'Toiletry'},
  {'id': 107, 'name': 'Keyboard'},
  {'id': 108, 'name': 'Tomato'},
  {'id': 109, 'name': 'Lantern'},
  {'id': 110, 'name': 'Machinery Vehicle'},
  {'id': 111, 'name': 'Fan'},
  {'id': 112, 'name': 'Green Vegetables'},
  {'id': 113, 'name': 'Banana'},
  {'id': 114, 'name': 'Baseball Glove'},
  {'id': 115, 'name': 'Airplane'},
  {'id': 116, 'name': 'Mouse'},
  {'id': 117, 'name': 'Train'},
  {'id': 118, 'name': 'Pumpkin'},
  {'id': 119, 'name': 'Soccer'},
  {'id': 120, 'name': 'Skiboard'},
  {'id': 121, 'name': 'Luggage'},
  {'id': 122, 'name': 'Nightstand'},
  {'id': 123, 'name': 'Teapot'},
  {'id': 124, 'name': 'Telephone'},
  {'id': 125, 'name': 'Trolley'},
  {'id': 126, 'name': 'Head Phone'},
  {'id': 127, 'name': 'Sports Car'},
  {'id': 128, 'name': 'Stop Sign'},
  {'id': 129, 'name': 'Dessert'},
  {'id': 130, 'name': 'Scooter'},
  {'id': 131, 'name': 'Stroller'},
  {'id': 132, 'name': 'Crane'},
  {'id': 133, 'name': 'Remote'},
  {'id': 134, 'name': 'Refrigerator'},
  {'id': 135, 'name': 'Oven'},
  {'id': 136, 'name': 'Lemon'},
  {'id': 137, 'name': 'Duck'},
  {'id': 138, 'name': 'Baseball Bat'},
  {'id': 139, 'name': 'Surveillance Camera'},
  {'id': 140, 'name': 'Cat'},
  {'id': 141, 'name': 'Jug'},
  {'id': 142, 'name': 'Broccoli'},
  {'id': 143, 'name': 'Piano'},
  {'id': 144, 'name': 'Pizza'},
  {'id': 145, 'name': 'Elephant'},
  {'id': 146, 'name': 'Skateboard'},
  {'id': 147, 'name': 'Surfboard'},
  {'id': 148, 'name': 'Gun'},
  {'id': 149, 'name': 'Skating and Skiing shoes'},
  {'id': 150, 'name': 'Gas stove'},
  {'id': 151, 'name': 'Donut'},
  {'id': 152, 'name': 'Bow Tie'},
  {'id': 153, 'name': 'Carrot'},
  {'id': 154, 'name': 'Toilet'},
  {'id': 155, 'name': 'Kite'},
  {'id': 156, 'name': 'Strawberry'},
  {'id': 157, 'name': 'Other Balls'},
  {'id': 158, 'name': 'Shovel'},
  {'id': 159, 'name': 'Pepper'},
  {'id': 160, 'name': 'Computer Box'},
  {'id': 161, 'name': 'Toilet Paper'},
  {'id': 162, 'name': 'Cleaning Products'},
  {'id': 163, 'name': 'Chopsticks'},
  {'id': 164, 'name': 'Microwave'},
  {'id': 165, 'name': 'Pigeon'},
  {'id': 166, 'name': 'Baseball'},
  {'id': 167, 'name': 'Cutting/chopping Board'},
  {'id': 168, 'name': 'Coffee Table'},
  {'id': 169, 'name': 'Side Table'},
  {'id': 170, 'name': 'Scissors'},
  {'id': 171, 'name': 'Marker'},
  {'id': 172, 'name': 'Pie'},
  {'id': 173, 'name': 'Ladder'},
  {'id': 174, 'name': 'Snowboard'},
  {'id': 175, 'name': 'Cookies'},
  {'id': 176, 'name': 'Radiator'},
  {'id': 177, 'name': 'Fire Hydrant'},
  {'id': 178, 'name': 'Basketball'},
  {'id': 179, 'name': 'Zebra'},
  {'id': 180, 'name': 'Grape'},
  {'id': 181, 'name': 'Giraffe'},
  {'id': 182, 'name': 'Potato'},
  {'id': 183, 'name': 'Sausage'},
  {'id': 184, 'name': 'Tricycle'},
  {'id': 185, 'name': 'Violin'},
  {'id': 186, 'name': 'Egg'},
  {'id': 187, 'name': 'Fire Extinguisher'},
  {'id': 188, 'name': 'Candy'},
  {'id': 189, 'name': 'Fire Truck'},
  {'id': 190, 'name': 'Billards'},
  {'id': 191, 'name': 'Converter'},
  {'id': 192, 'name': 'Bathtub'},
  {'id': 193, 'name': 'Wheelchair'},
  {'id': 194, 'name': 'Golf Club'},
  {'id': 195, 'name': 'Briefcase'},
  {'id': 196, 'name': 'Cucumber'},
  {'id': 197, 'name': 'Cigar/Cigarette '},
  {'id': 198, 'name': 'Paint Brush'},
  {'id': 199, 'name': 'Pear'},
  {'id': 200, 'name': 'Heavy Truck'},
  {'id': 201, 'name': 'Hamburger'},
  {'id': 202, 'name': 'Extractor'},
  {'id': 203, 'name': 'Extension Cord'},
  {'id': 204, 'name': 'Tong'},
  {'id': 205, 'name': 'Tennis Racket'},
  {'id': 206, 'name': 'Folder'},
  {'id': 207, 'name': 'American Football'},
  {'id': 208, 'name': 'earphone'},
  {'id': 209, 'name': 'Mask'},
  {'id': 210, 'name': 'Kettle'},
  {'id': 211, 'name': 'Tennis'},
  {'id': 212, 'name': 'Ship'},
  {'id': 213, 'name': 'Swing'},
  {'id': 214, 'name': 'Coffee Machine'},
  {'id': 215, 'name': 'Slide'},
  {'id': 216, 'name': 'Carriage'},
  {'id': 217, 'name': 'Onion'},
  {'id': 218, 'name': 'Green beans'},
  {'id': 219, 'name': 'Projector'},
  {'id': 220, 'name': 'Frisbee'},
  {'id': 221, 'name': 'Washing Machine/Drying Machine'},
  {'id': 222, 'name': 'Chicken'},
  {'id': 223, 'name': 'Printer'},
  {'id': 224, 'name': 'Watermelon'},
  {'id': 225, 'name': 'Saxophone'},
  {'id': 226, 'name': 'Tissue'},
  {'id': 227, 'name': 'Toothbrush'},
  {'id': 228, 'name': 'Ice cream'},
  {'id': 229, 'name': 'Hot air balloon'},
  {'id': 230, 'name': 'Cello'},
  {'id': 231, 'name': 'French Fries'},
  {'id': 232, 'name': 'Scale'},
  {'id': 233, 'name': 'Trophy'},
  {'id': 234, 'name': 'Cabbage'},
  {'id': 235, 'name': 'Hot dog'},
  {'id': 236, 'name': 'Blender'},
  {'id': 237, 'name': 'Peach'},
  {'id': 238, 'name': 'Rice'},
  {'id': 239, 'name': 'Wallet/Purse'},
  {'id': 240, 'name': 'Volleyball'},
  {'id': 241, 'name': 'Deer'},
  {'id': 242, 'name': 'Goose'},
  {'id': 243, 'name': 'Tape'},
  {'id': 244, 'name': 'Tablet'},
  {'id': 245, 'name': 'Cosmetics'},
  {'id': 246, 'name': 'Trumpet'},
  {'id': 247, 'name': 'Pineapple'},
  {'id': 248, 'name': 'Golf Ball'},
  {'id': 249, 'name': 'Ambulance'},
  {'id': 250, 'name': 'Parking meter'},
  {'id': 251, 'name': 'Mango'},
  {'id': 252, 'name': 'Key'},
  {'id': 253, 'name': 'Hurdle'},
  {'id': 254, 'name': 'Fishing Rod'},
  {'id': 255, 'name': 'Medal'},
  {'id': 256, 'name': 'Flute'},
  {'id': 257, 'name': 'Brush'},
  {'id': 258, 'name': 'Penguin'},
  {'id': 259, 'name': 'Megaphone'},
  {'id': 260, 'name': 'Corn'},
  {'id': 261, 'name': 'Lettuce'},
  {'id': 262, 'name': 'Garlic'},
  {'id': 263, 'name': 'Swan'},
  {'id': 264, 'name': 'Helicopter'},
  {'id': 265, 'name': 'Green Onion'},
  {'id': 266, 'name': 'Sandwich'},
  {'id': 267, 'name': 'Nuts'},
  {'id': 268, 'name': 'Speed Limit Sign'},
  {'id': 269, 'name': 'Induction Cooker'},
  {'id': 270, 'name': 'Broom'},
  {'id': 271, 'name': 'Trombone'},
  {'id': 272, 'name': 'Plum'},
  {'id': 273, 'name': 'Rickshaw'},
  {'id': 274, 'name': 'Goldfish'},
  {'id': 275, 'name': 'Kiwi fruit'},
  {'id': 276, 'name': 'Router/modem'},
  {'id': 277, 'name': 'Poker Card'},
  {'id': 278, 'name': 'Toaster'},
  {'id': 279, 'name': 'Shrimp'},
  {'id': 280, 'name': 'Sushi'},
  {'id': 281, 'name': 'Cheese'},
  {'id': 282, 'name': 'Notepaper'},
  {'id': 283, 'name': 'Cherry'},
  {'id': 284, 'name': 'Pliers'},
  {'id': 285, 'name': 'CD'},
  {'id': 286, 'name': 'Pasta'},
  {'id': 287, 'name': 'Hammer'},
  {'id': 288, 'name': 'Cue'},
  {'id': 289, 'name': 'Avocado'},
  {'id': 290, 'name': 'Hami melon'},
  {'id': 291, 'name': 'Flask'},
  {'id': 292, 'name': 'Mushroom'},
  {'id': 293, 'name': 'Screwdriver'},
  {'id': 294, 'name': 'Soap'},
  {'id': 295, 'name': 'Recorder'},
  {'id': 296, 'name': 'Bear'},
  {'id': 297, 'name': 'Eggplant'},
  {'id': 298, 'name': 'Board Eraser'},
  {'id': 299, 'name': 'Coconut'},
  {'id': 300, 'name': 'Tape Measure/ Ruler'},
  {'id': 301, 'name': 'Pig'},
  {'id': 302, 'name': 'Showerhead'},
  {'id': 303, 'name': 'Globe'},
  {'id': 304, 'name': 'Chips'},
  {'id': 305, 'name': 'Steak'},
  {'id': 306, 'name': 'Crosswalk Sign'},
  {'id': 307, 'name': 'Stapler'},
  {'id': 308, 'name': 'Camel'},
  {'id': 309, 'name': 'Formula 1 '},
  {'id': 310, 'name': 'Pomegranate'},
  {'id': 311, 'name': 'Dishwasher'},
  {'id': 312, 'name': 'Crab'},
  {'id': 313, 'name': 'Hoverboard'},
  {'id': 314, 'name': 'Meatball'},
  {'id': 315, 'name': 'Rice Cooker'},
  {'id': 316, 'name': 'Tuba'},
  {'id': 317, 'name': 'Calculator'},
  {'id': 318, 'name': 'Papaya'},
  {'id': 319, 'name': 'Antelope'},
  {'id': 320, 'name': 'Parrot'},
  {'id': 321, 'name': 'Seal'},
  {'id': 322, 'name': 'Butterfly'},
  {'id': 323, 'name': 'Dumbbell'},
  {'id': 324, 'name': 'Donkey'},
  {'id': 325, 'name': 'Lion'},
  {'id': 326, 'name': 'Urinal'},
  {'id': 327, 'name': 'Dolphin'},
  {'id': 328, 'name': 'Electric Drill'},
  {'id': 329, 'name': 'Hair Dryer'},
  {'id': 330, 'name': 'Egg tart'},
  {'id': 331, 'name': 'Jellyfish'},
  {'id': 332, 'name': 'Treadmill'},
  {'id': 333, 'name': 'Lighter'},
  {'id': 334, 'name': 'Grapefruit'},
  {'id': 335, 'name': 'Game board'},
  {'id': 336, 'name': 'Mop'},
  {'id': 337, 'name': 'Radish'},
  {'id': 338, 'name': 'Baozi'},
  {'id': 339, 'name': 'Target'},
  {'id': 340, 'name': 'French'},
  {'id': 341, 'name': 'Spring Rolls'},
  {'id': 342, 'name': 'Monkey'},
  {'id': 343, 'name': 'Rabbit'},
  {'id': 344, 'name': 'Pencil Case'},
  {'id': 345, 'name': 'Yak'},
  {'id': 346, 'name': 'Red Cabbage'},
  {'id': 347, 'name': 'Binoculars'},
  {'id': 348, 'name': 'Asparagus'},
  {'id': 349, 'name': 'Barbell'},
  {'id': 350, 'name': 'Scallop'},
  {'id': 351, 'name': 'Noddles'},
  {'id': 352, 'name': 'Comb'},
  {'id': 353, 'name': 'Dumpling'},
  {'id': 354, 'name': 'Oyster'},
  {'id': 355, 'name': 'Table Tennis paddle'},
  {'id': 356, 'name': 'Cosmetics Brush/Eyeliner Pencil'},
  {'id': 357, 'name': 'Chainsaw'},
  {'id': 358, 'name': 'Eraser'},
  {'id': 359, 'name': 'Lobster'},
  {'id': 360, 'name': 'Durian'},
  {'id': 361, 'name': 'Okra'},
  {'id': 362, 'name': 'Lipstick'},
  {'id': 363, 'name': 'Cosmetics Mirror'},
  {'id': 364, 'name': 'Curling'},
  {'id': 365, 'name': 'Table Tennis '},
]


def _get_builtin_metadata():
    id_to_name = {x['id']: x['name'] for x in categories_v2_fix}
    thing_dataset_id_to_contiguous_id = {
        x['id']: i for i, x in enumerate(
            sorted(categories_v2_fix, key=lambda x: x['id']))}
    thing_classes = [id_to_name[k] for k in sorted(id_to_name)]
    return {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes}


_PREDEFINED_SPLITS_OBJECTS365 = {
    "objects365_v2_train": ("objects365/images/train", "objects365/annotations/zhiyuan_objv2_train_fixname_fixmiss.json"),
    # 80,000 images, 1,240,587 annotations
    "objects365_v2_val": ("objects365/images/val", "objects365/annotations/zhiyuan_objv2_val_fixname_replace.json"),
    "objects365_v2_val_rare": ("objects365/images/val", "objects365/annotations/zhiyuan_objv2_val_fixname_rare.json"),
}

_root = os.getenv("DETECTRON2_DATASETS", "datasets")

for key, (image_root, json_file) in _PREDEFINED_SPLITS_OBJECTS365.items():
    register_coco_instances(
        key,
        _get_builtin_metadata(),
        os.path.join(_root, json_file) if "://" not in json_file else json_file,
        os.path.join(_root, image_root),
    )