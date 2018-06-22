# -*- coding: utf-8 -*-
"""
Created on Fri May 18 12:58:34 2018

@author: deepe
"""

import facebook as fb
access_token = "EAACEdEose0cBAC8mNfJ7KVw0WN0lSpHvByWjMMVZBfl4d67xTxrKGO9TXQukonuQ2oHCWnfyslgMdw42HvUGfvifTgAsNZBVJbwAbMZAFNJIUcNRsgdSZByUL32VNWv1JZByDAd0QRAEULb29oSEJZBEPspI0N2w5Q8CAvSgYP7rkqPVkHYNWpMVtGSGsSZBQgxlWAhZC6hNWAZDZD"
status = "Uploaded from Python Code using Graph API"
graph = fb.GraphAPI(access_token)
post_id = graph.put_wall_post(status)
post_id1 = graph.put_photo(image= open('22117648_2018197925076218_1151300939_o.jpg', 'rb'),message = 'Graph API Explorer')