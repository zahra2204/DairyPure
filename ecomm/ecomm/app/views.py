# import smtplib
import os
import numpy as np
import pandas as pd
import razorpay
from django.conf import settings
from django.contrib.auth.decorators import login_required
from django.db.models import Q
from django.http import JsonResponse
from django.shortcuts import render, redirect
from django.utils.decorators import method_decorator
from django.views import View
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import json
from .models import Product, Customer, Cart, Payment, OrderPlaced, Wishlist, Contact
from .forms import CustomerRegistrationForm, CustomerProfileForm, RecipeRecommendationForm
from django.contrib import messages
import logging
import google.generativeai as genai
# Create your views here.

@login_required
def home(request):
    totalitem = 0
    wishitem = 0
    if request.user.is_authenticated:
        totalitem = len(Cart.objects.filter(user=request.user))
        wishitem = len(Wishlist.objects.filter(user=request.user))
    return render(request, "app/home.html", locals())


@login_required
def about(request):
    totalitem = 0
    wishitem = 0
    if request.user.is_authenticated:
        totalitem = len(Cart.objects.filter(user=request.user))
        wishitem = len(Wishlist.objects.filter(user=request.user))
    return render(request, "app/about.html", locals())


@login_required
def Learn_more(request):
    totalitem = 0
    wishlist = 0
    if request.user.is_authenticated:
        totalitem = len(Cart.objects.filter(user=request.user))
        wishitem = len(Wishlist.objects.filter(user=request.user))
    return render(request, 'app/Learn_more.html', locals())


@login_required
def contact(request):
    totalitem = 0
    wishitem = 0
    if request.user.is_authenticated:
        totalitem = len(Cart.objects.filter(user=request.user))
        wishitem = len(Wishlist.objects.filter(user=request.user))
    thank = False
    if request.method == "POST":
        name = request.POST.get('name', '')
        email = request.POST.get('email', '')
        phone = request.POST.get('phone', '')
        desc = request.POST.get('desc', '')
        contact = Contact(name=name, email=email, phone=phone, desc=desc)
        contact.save()
        thank = True
    return render(request, "app/contact.html", locals())


@method_decorator(login_required, name='dispatch')
class CategoryView(View):
    def get(self, request, val):
        totalitem = 0
        wishitem = 0
        if request.user.is_authenticated:
            totalitem = len(Cart.objects.filter(user=request.user))
            wishitem = len(Wishlist.objects.filter(user=request.user))
        product = Product.objects.filter(category=val)
        title = Product.objects.filter(category=val).values('title')
        return render(request, 'app/category.html', locals())


@method_decorator(login_required, name='dispatch')
class CategoryTitle(View):
    def get(self, request, val):
        product = Product.objects.filter(title=val)
        title = Product.objects.filter(category=product[0].category).values('title')
        totalitem = 0
        wishitem = 0
        if request.user.is_authenticated:
            totalitem = len(Cart.objects.filter(user=request.user))
            wishitem = len(Wishlist.objects.filter(user=request.user))
        return render(request, 'app/category.html', locals())


@method_decorator(login_required, name='dispatch')
class ProductDetail(View):
    def get(self, request, pk):
        product = Product.objects.get(pk=pk)
        related_products = Product.objects.filter(category=product.category).exclude(pk=pk)
        wishlist = Wishlist.objects.filter(Q(product=product) & Q(user=request.user))
        totalitem = 0
        wishitem = 0
        if request.user.is_authenticated:
            totalitem = len(Cart.objects.filter(user=request.user))
            wishitem = len(Wishlist.objects.filter(user=request.user))
        return render(request, 'app/productdetail.html', locals())


class CustomerRegistrationView(View):
    def get(self, request):
        form = CustomerRegistrationForm()
        return render(request, 'app/customerregistration.html', locals())

    def post(self, request):
        form = CustomerRegistrationForm(request.POST)
        if form.is_valid():
            form.save()
            messages.success(request, "User Registered Successfully")
        else:
            messages.warning(request, "Invalid Input data")
        return render(request, 'app/customerregistration.html', locals())


@method_decorator(login_required, name='dispatch')
class ProfileView(View):
    def get(self, request):
        form = CustomerProfileForm()
        totalitem = 0
        wishitem = 0
        if request.user.is_authenticated:
            totalitem = len(Cart.objects.filter(user=request.user))
            wishitem = len(Wishlist.objects.filter(user=request.user))
        return render(request, 'app/profile.html', locals())

    def post(self, request):
        form = CustomerProfileForm(request.POST)
        if form.is_valid():
            user = request.user
            name = form.cleaned_data['name']
            locality = form.cleaned_data['locality']
            city = form.cleaned_data['city']
            mobile = form.cleaned_data['mobile']
            state = form.cleaned_data['state']
            zipcode = form.cleaned_data['zipcode']

            reg = Customer(user=user, name=name, locality=locality, mobile=mobile, city=city, state=state, zipcode=zipcode)
            reg.save()
            messages.success(request, 'Profile saved successfully!')
        else:
            messages.warning(request, 'Invalid input data')
        return render(request, 'app/profile.html', locals())


@login_required
def address(request):
    add = Customer.objects.filter(user=request.user)
    totalitem = 0
    wishitem = 0
    if request.user.is_authenticated:
        totalitem = len(Cart.objects.filter(user=request.user))
        wishitem = len(Wishlist.objects.filter(user=request.user))
    return render(request, 'app/address.html', locals())


@method_decorator(login_required, name='dispatch')
class updateAddress(View):
    def get(self, request, pk):
        add = Customer.objects.get(pk=pk)
        form = CustomerProfileForm(instance=add)
        totalitem = 0
        wishitem = 0
        if request.user.is_authenticated:
            totalitem = len(Cart.objects.filter(user=request.user))
            wishitem = len(Wishlist.objects.filter(user=request.user))
        return render(request, 'app/updateAddress.html', locals())

    def post(self, request, pk):
        form = CustomerProfileForm(request.POST)
        if form.is_valid():
            add = Customer.objects.get(pk=pk)
            add.name = form.cleaned_data['name']
            add.locality = form.cleaned_data['locality']
            add.city = form.cleaned_data['city']
            add.mobile = form.cleaned_data['mobile']
            add.state = form.cleaned_data['state']
            add.zipcode = form.cleaned_data['zipcode']
            add.save()
            messages.success(request, "Profile Updated Successfully")
        else:
            messages.warning(request, "Invalid Input Data")
        return redirect("address")


@login_required
def add_to_cart(request):
    user = request.user
    product_id = request.GET.get('prod_id')
    product = Product.objects.get(id=product_id)
    Cart(user=user, product=product).save()
    return redirect('/cart')


@login_required
def show_cart(request):
    user = request.user
    cart = Cart.objects.filter(user=user)
    amount = 0
    for p in cart:
        value = p.quantity * p.product.discounted_price
        amount = amount + value
    totalamount = amount + 40
    totalitem = 0
    wishitem = 0
    if request.user.is_authenticated:
        totalitem = len(Cart.objects.filter(user=request.user))
        wishitem = len(Wishlist.objects.filter(user=request.user))
    return render(request, 'app/addtocart.html', locals())


@login_required
def plus_cart(request):
    if request.method == 'GET':
        prod_id = request.GET['prod_id']
        # incase JS kaam nhi kare toh plus and minus dono functions me
        # "c = Cart.objects.get" yaha pe get ki jagah filter lagana and wapas filter hata k get lagana and restart the server.
        c = Cart.objects.get(Q(product=prod_id) & Q(user=request.user))
        c.quantity += 1
        c.save()
        user = request.user
        cart = Cart.objects.filter(user=user)
        amount = 0
        for p in cart:
            value = p.quantity * p.product.discounted_price
            amount = amount + value
        totalamount = amount + 40
        data = {
            'quantity': c.quantity,
            'amount': amount,
            'totalamount': totalamount
        }
        return JsonResponse(data)


@login_required
def minus_cart(request):
    if request.method == 'GET':
        prod_id = request.GET['prod_id']
        c = Cart.objects.get(Q(product=prod_id) & Q(user=request.user))
        c.quantity -= 1
        c.save()
        user = request.user
        cart = Cart.objects.filter(user=user)
        amount = 0
        for p in cart:
            value = p.quantity * p.product.discounted_price
            amount = amount + value
        totalamount = amount + 40
        data = {
            'quantity': c.quantity,
            'amount': amount,
            'totalamount': totalamount
        }
        return JsonResponse(data)


@login_required
def remove_cart(request):
    if request.method == 'GET':
        prod_id = request.GET['prod_id']
        c = Cart.objects.filter(Q(product=prod_id) & Q(user=request.user))
        c.delete()
        user = request.user
        cart = Cart.objects.filter(user=user)
        amount = 0
        for p in cart:
            value = p.quantity * p.product.discounted_price
            amount = amount + value
        totalamount = amount + 40
        data = {
            'amount': amount,
            'totalamount': totalamount
        }
        return JsonResponse(data)


@method_decorator(login_required, name='dispatch')
class checkout(View):
    def get(self, request):
        totalitem = 0
        wishitem = 0
        if request.user.is_authenticated:
            totalitem = len(Cart.objects.filter(user=request.user))
            wishitem = len(Wishlist.objects.filter(user=request.user))
        user = request.user
        add = Customer.objects.filter(user=user)
        cart_items = Cart.objects.filter(user=user)
        amount = 0
        for p in cart_items:
            value = p.quantity * p.product.discounted_price
            amount = amount + value
        totalamount = amount + 40
        razoramount = int(totalamount * 100)
        client = razorpay.Client(auth=(settings.RAZOR_KEY_ID, settings.RAZOR_KEY_SECRET))
        data = {'amount': razoramount, 'currency': 'INR', 'receipt': 'order_rcptid_11'}
        payment_response = client.order.create(data=data)
        print(payment_response)
        # {'amount': 8500, 'amount_due': 8500, 'amount_paid': 0, 'attempts': 0, 'created_at': 1737884210, 'currency': 'INR', 'entity': 'order', 'id': 'order_Po1bJqA1Aam3F4', 'notes': [], 'offer_id': None, 'receipt': 'order_rcptid_11', 'status': 'created'}
        order_id = payment_response['id']
        order_status = payment_response['status']
        if order_status == 'created':
            payment = Payment(
                user=user,
                amount=totalamount,
                razorpay_order_id=order_id,
                razorpay_payment_status=order_status
            )
            payment.save()
        return render(request, 'app/checkout.html', locals())


@login_required
def payment_done(request):
    order_id = request.GET.get('order_id')
    payment_id = request.GET.get('payment_id')
    cust_id = request.GET.get('cust_id')
    # print("payment_done : oid = ",order_id," pid = ",payment_id, " cid = ",cust_id)
    user = request.user
    # return redirect('orders')
    customer = Customer.objects.get(id=cust_id)
    # To update payment status and payment id
    payment = Payment.objects.get(razorpay_order_id=order_id)
    payment.paid = True
    payment.razorpay_payment_id = payment_id
    payment.save()
    # To save order details
    cart = Cart.objects.filter(user=user)
    for c in cart:
        OrderPlaced(user=user, customer=customer, product=c.product, quantity=c.quantity, payment=payment).save()
        c.delete()
    return redirect('orders')


@login_required
def orders(request):
    totalitem = 0
    wishitem = 0
    if request.user.is_authenticated:
        totalitem = len(Cart.objects.filter(user=request.user))
        wishitem = len(Wishlist.objects.filter(user=request.user))
    order_placed = OrderPlaced.objects.filter(user=request.user)
    return render(request, 'app/orders.html', locals())


@login_required
def show_wishlist(request):
    user = request.user
    totalitem = 0
    wishitem = 0
    if request.user.is_authenticated:
        totalitem = len(Cart.objects.filter(user=request.user))
        wishitem = len(Wishlist.objects.filter(user=request.user))
    product = Wishlist.objects.filter(user=user)
    return render(request, 'app/wishlist.html', locals())


@login_required
def plus_wishlist(request):
    if request.method == 'GET':
        prod_id = request.GET['prod_id']
        product = Product.objects.get(id=prod_id)
        user = request.user
        Wishlist(user=user, product=product).save()
        data = {
            'message': 'Wishlist added Successfully',
        }
        return JsonResponse(data)


@login_required
def minus_wishlist(request):
    if request.method == 'GET':
        prod_id = request.GET['prod_id']
        product = Product.objects.get(id=prod_id)
        user = request.user
        Wishlist.objects.filter(user=user, product=product).delete()
        data = {
            'message': 'Wishlist removed Successfully',
        }
        return JsonResponse(data)


@login_required
def search(request):
    query = request.GET['search']
    product = Product.objects.filter(Q(title__icontains=query))
    totalitem = 0
    wishitem = 0
    if request.user.is_authenticated:
        totalitem = len(Cart.objects.filter(user=request.user))
        wishitem = len(Wishlist.objects.filter(user=request.user))
    return render(request, 'app/search.html', locals())



# Load dataset
dataset_path = os.path.join(os.path.dirname(__file__), '..', 'media','product', 'dairy_food_nutrition_dataset.csv') # Update with correct path
data = pd.read_csv(dataset_path)

# Preprocessing
scaler = StandardScaler()
X_numerical = scaler.fit_transform(data[['Calories', 'Protein', 'Carbohydrates', 'Fat', 'Sugars']])
X_combined = np.hstack([X_numerical])
knn = NearestNeighbors(n_neighbors=3, metric='euclidean')
knn.fit(X_combined)



def recommend_recipes(input_features):
    try:
        input_features_scaled = scaler.transform([input_features])
        distances, indices = knn.kneighbors(input_features_scaled)
        recommendations = data.iloc[indices[0]]
        return recommendations
    except Exception as e:
        print(f"Error: {e}")
        return None

def recommendations(request):
    if request.method == 'POST':
        form = RecipeRecommendationForm(request.POST)
        if form.is_valid():
            input_features = [
                form.cleaned_data['Calories'],
                form.cleaned_data['Protein'],
                form.cleaned_data['Carbohydrates'],
                form.cleaned_data['Fat'],
                form.cleaned_data['Sugars'],
            ]
            recommendations = recommend_recipes(input_features)
            if recommendations is not None:
                return render(request, 'app/recommendations.html', {
                    'recommendations': recommendations.to_dict(orient='records'),
                })
            else:
                return render(request, 'app/recommendations.html', {
                    'recommendations': [],
                    'error': "Error generating recommendations."
                })
        else:
            return render(request, 'app/recommendations.html', {
                'form': form,
                'recommendations': [],
                'error': "Please ensure all fields are filled with valid numbers."
            })

    form = RecipeRecommendationForm()
    return render(request, 'app/recommendations.html', {
        'form': form,
        'recommendations': [],
        'error': ""
    })


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure Gemini API
API_KEY = "AIzaSyDtrfi5aYctFGhrp_WlB1LggX_frbVjni0"
genai.configure(api_key=API_KEY)

# Initialize the model - using gemini-1.5-flash
try:
    model = genai.GenerativeModel("gemini-1.5-flash")
except Exception as e:
    logger.error(f"Failed to initialize Gemini model: {e}")
    model = None

# Allowed topics
allowed_topics = ["account", "login", "password", "order", "checkout", "payment", "subscription", "delivery", "refund",
                  "return", "website", "customer support", "product", "help"]


def is_relevant_query(user_query):
    """Check if the user query contains any of the allowed topics."""
    if not user_query:
        return False
    return any(topic in user_query.lower() for topic in allowed_topics)


def get_products():
    """Retrieve product details from the database in a structured list format."""
    products = Product.objects.all()  # Fetch products
    logger.info(f"Fetched products: {products}")  # Debugging info

    if not products:
        return []

    product_list = []
    for product in products:
        logger.info(f"Processing product: {product._dict_}")  # Log product data
        product_list.append({
            "title": product.title,
            "price": f"{product.discounted_price}",
            "details": product.MFD
        })

    return product_list

@login_required
def get_previous_orders(user):
    orders = OrderPlaced.objects.filter(user=user).order_by('-ordered_date')

    if not orders.exists():
        return "You have no previous orders."

    order_list = []
    for i, order in enumerate(orders):
        # Fetch the related payment record correctly
        payment = Payment.objects.filter(orderplaced=order).first()  # Correct field name

        # Get payment amount or display "N/A" if no payment found
        amount = payment.amount if payment else "N/A"

        order_list.append(f"{i+1}. Order #{order.id} {order.product} (Qty:{order.quantity}) - {order.status} - Rs.{amount} - Payment Status:{order.payment.paid}")

    return f"Here are your previous orders:\n" + "\n".join(order_list)



@csrf_exempt
@require_http_methods(["POST", "OPTIONS"])  # Allow POST and OPTIONS methods
def chatbot_endpoint(request):
    """Handle chatbot API requests."""

    # Handle OPTIONS preflight request
    if request.method == "OPTIONS":
        response = JsonResponse({"message": "CORS preflight successful."})
        response["Access-Control-Allow-Origin"] = "*"
        response["Access-Control-Allow-Methods"] = "POST, OPTIONS"
        response["Access-Control-Allow-Headers"] = "Content-Type"
        return response

    try:
        data = json.loads(request.body)
        user_query = data.get("message", "").strip()

        logger.info(f"Received query: {user_query}")

        if not user_query:
            return JsonResponse({"response": "Please enter a question or message."})

        if not is_relevant_query(user_query):
            return JsonResponse({
                                    "response": "I can only help with questions related to our website, products, orders, account, and customer support. How can I assist you with these topics?"})

        # Handle product-related queries
        if "product" in user_query.lower():
            products = get_products()  # Fetch products from DB

            if not products:
                return JsonResponse({"response": "No products available at the moment."})

            # Format products as a numbered list
            product_list_text = "\n".join(
                [f"{i + 1}. {product['title']} - Price: Rs.{product['price']}\n   Mfg.{product['details']}\n" for i, product in enumerate(products)]
            )

            response_text = f"Here are some available products:\n\n{product_list_text}"

            return JsonResponse({"response": response_text}, headers={"Access-Control-Allow-Origin": "*"})

        if "order" in user_query.lower():
            user = request.user  # No need for user.user

            if not user.is_authenticated:
                return JsonResponse({"response": "You need to log in to check your orders."})

            # Get the user's orders
            orders = OrderPlaced.objects.filter(user=user).order_by('-ordered_date')

            if not orders.exists():
                return JsonResponse({"response": "You have no previous orders."})

            # Create a numbered list of orders with correct payment details
            order_list = []
            for i, order in enumerate(orders):
                # Fetch the payment associated with the order using the correct field
                payment = Payment.objects.filter(orderplaced=order).first()  # Correct field usage

                # If payment exists, get the amount; otherwise, show "N/A"
                amount = payment.amount if payment else "N/A"

                order_list.append(
                    f"{i + 1}. Order #{order.id} - {order.product} (Qty: {order.quantity}) - {order.status} - Rs.{amount} - Payment Status:{order.payment.paid}"
                )

            return JsonResponse(
                {"response": f"Here are your previous orders:" + " ".join(order_list)},
                headers={"Access-Control-Allow-Origin": "*"}
            )

        if "account" in user_query.lower() or "payment" in user_query.lower():
            user = request.user  # Fix user object usage

            if not user.is_authenticated:
                return JsonResponse({"response": "You need to log in to check your account details."})

            # Fetch last order and payment status
            last_order = OrderPlaced.objects.filter(user=user).order_by('-ordered_date').first()
            order_status = last_order.status if last_order else "No payments found"

            user_info = (
                f"---User Information:---\n"
                f"Name: {user} |\n"
                f"Email: {user.email} |\n"
                f"Last Order Status: {order_status}"
            )

            return JsonResponse({"response": user_info}, headers={"Access-Control-Allow-Origin": "*"})

        # Generate response using Gemini AI
        if model is None:
            return JsonResponse({
                                    "response": "The AI service is currently unavailable. Please try again later or contact customer support."})

        try:
            context = "You are a helpful customer service chatbot for an e-commerce website."
            response = model.generate_content(f"{context}\n\nCustomer query: {user_query}")

            ai_response = response.text.strip() if hasattr(response, "text") else "I'm sorry, but I couldn't generate a response."

            logger.info(f"Generated response for query: {user_query}")

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            ai_response = "Sorry, I'm having trouble responding right now. Please try again later."

        return JsonResponse({"response": ai_response}, headers={"Access-Control-Allow-Origin": "*"})

    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON input."}, status=400)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return JsonResponse({"error": "An unexpected error occurred."}, status=500)
# your_email = "fashionhubshopping112204@gmail.com"
# your_app_password = "zaud kdzf bhqd fbep"
#
#
# def send_email(name, email, body):                    # Email sending code starts here
#     try:
#         smtp_obj = smtplib.SMTP('smtp.gmail.com', 587)
#         smtp_obj.ehlo()
#         smtp_obj.starttls()
#         smtp_obj.login(your_email, your_app_password)
#         smtp_obj.sendmail(your_email, email, body.encode())
#         print('Email sent successfully to %s' % email)
#         return True
#     except Exception as e:
#         print(f"Error sending email to {email}: {e}")
#         return False
#     finally:
#         smtp_obj.quit()