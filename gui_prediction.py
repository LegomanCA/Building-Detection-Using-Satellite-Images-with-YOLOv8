import os
import logging
import webbrowser
import tkinter as tk
from tkinter import filedialog, Label, Button, Frame, Listbox, Scrollbar, Canvas
from PIL import Image, ImageTk
from ultralytics import YOLO
import cv2

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class YoloApp:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLOv8 Building Detection")
        self.center_window(1200, 800)  # Center the GUI window
        self.model = YOLO('runs/detect/train_experiment3/weights/best.pt')  # Load the best trained model
        self.image_paths = []
        self.raw_images = {}
        self.predicted_images = {}
        self.split_images_map = {}
        self.current_image_path = None
        self.showing_raw = False
        self.zoom_scale = 1.0
        self.max_zoom = 2.0  # Five times from default
        self.min_zoom = 0.2  # Five times zoomed out from default

        self.setup_ui()

    def center_window(self, width, height):
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        x = (screen_width // 2) - (width // 2)
        y = (screen_height // 2) - (height // 2)
        self.root.geometry(f'{width}x{height}+{x}+{y}')

    def setup_ui(self):
        # Create main frames
        self.image_frame = Frame(self.root, bg='#FFD580', width=800, height=800)
        self.image_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.sidebar = Frame(self.root, width=400, padx=20, pady=20)
        self.sidebar.pack(side=tk.RIGHT, fill=tk.Y)

        # Canvas for image display
        self.canvas = Canvas(self.image_frame, bg='#FFD580')
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Display an empty image initially to avoid blank space
        empty_image = Image.new('RGB', (800, 800), color='#FFD580')
        self.display_image(empty_image)

        # Buttons and labels
        button_style = {'bg': '#4CAF50', 'fg': 'white', 'font': ('Helvetica', 12, 'bold'), 'relief': 'raised', 'bd': 5}
        self.upload_button = Button(self.sidebar, text="Upload Image", command=self.upload_image, **button_style)
        self.upload_button.pack(pady=10)

        self.switch_button = Button(self.sidebar, text="Switch Image", command=self.switch_image, state=tk.DISABLED, **button_style)
        self.switch_button.pack(pady=10)

        self.building_count_label = Label(self.sidebar, text="Buildings Detected: 0", font=('Helvetica', 14, 'bold'), pady=10)
        self.building_count_label.pack()

        # Thumbnail listbox with scrollbar
        self.thumbnail_frame = Frame(self.sidebar)
        self.thumbnail_frame.pack(fill=tk.BOTH, expand=True)

        self.thumbnail_scrollbar = Scrollbar(self.thumbnail_frame, orient=tk.VERTICAL)
        self.thumbnail_listbox = Listbox(self.thumbnail_frame, yscrollcommand=self.thumbnail_scrollbar.set, height=10, font=('Helvetica', 12))
        self.thumbnail_scrollbar.config(command=self.thumbnail_listbox.yview)
        self.thumbnail_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.thumbnail_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.thumbnail_listbox.bind('<Double-1>', self.on_thumbnail_double_click)

        self.canvas.bind('<MouseWheel>', self.on_mouse_wheel)  # For zooming with the mouse wheel
        self.canvas.bind('<Button-4>', self.on_mouse_wheel)    # For Linux
        self.canvas.bind('<Button-5>', self.on_mouse_wheel)    # For Linux

        self.canvas.bind('<ButtonPress-1>', self.pan_start)
        self.canvas.bind('<B1-Motion>', self.pan_move)

        # Zoom buttons
        zoom_button_style = {'bg': '#4CAF50', 'fg': 'white', 'font': ('Helvetica', 12, 'bold'), 'relief': 'raised', 'bd': 5}
        self.zoom_frame = Frame(self.image_frame, bg='#FFD580')
        self.zoom_frame.pack(side=tk.BOTTOM, anchor='se', padx=20, pady=20)

        zoom_in_icon_path = 'src/zoom_in_icon.png'
        zoom_out_icon_path = 'src/zoom_out_icon.png'
        
        zoom_in_icon = ImageTk.PhotoImage(Image.open(zoom_in_icon_path).resize((30, 30), Image.Resampling.LANCZOS))
        zoom_out_icon = ImageTk.PhotoImage(Image.open(zoom_out_icon_path).resize((30, 30), Image.Resampling.LANCZOS))

        self.zoom_in_button = Button(self.zoom_frame, image=zoom_in_icon, command=self.zoom_in, **zoom_button_style)
        self.zoom_in_button.image = zoom_in_icon  # Keep a reference to avoid garbage collection
        self.zoom_in_button.pack(side=tk.LEFT, padx=5)

        self.zoom_out_button = Button(self.zoom_frame, image=zoom_out_icon, command=self.zoom_out, **zoom_button_style)
        self.zoom_out_button.image = zoom_out_icon  # Keep a reference to avoid garbage collection
        self.zoom_out_button.pack(side=tk.RIGHT, padx=5)

        # Exit and Credits buttons
        exit_button_style = {'bg': '#F44336', 'fg': 'white', 'font': ('Helvetica', 12, 'bold'), 'relief': 'raised', 'bd': 5}
        self.exit_button = Button(self.sidebar, text="EXIT", command=self.root.quit, **exit_button_style)
        self.exit_button.pack(side=tk.BOTTOM, pady=10)

        credits_button_style = {'bg': '#DDA0DD', 'fg': 'white', 'font': ('Helvetica', 12, 'bold'), 'relief': 'raised', 'bd': 5}
        self.credits_button = Button(self.sidebar, text="CREDITS", command=self.show_credits, **credits_button_style)
        self.credits_button.pack(side=tk.BOTTOM, pady=10)

    def show_credits(self):
        self.hide_main_ui()

        # Credits Panel
        self.credits_frame = Frame(self.root, bg='#FFF8DC', width=1200, height=800)
        self.credits_frame.pack(fill=tk.BOTH, expand=True)

        credit_image_path = 'src/credit.png'
        credit_image = ImageTk.PhotoImage(Image.open(credit_image_path).resize((400, 200), Image.Resampling.LANCZOS))
        credit_image_label = Label(self.credits_frame, image=credit_image, bg='#FFF8DC')
        credit_image_label.image = credit_image  # Keep a reference to avoid garbage collection
        credit_image_label.pack(pady=(50, 20))

        header_label = Label(self.credits_frame, text="BU PROJE TÜBİTAK 2209-A KAPSAMINDA GELİŞTİRİLMEKTEDİR.", font=('Helvetica', 14, 'bold'), bg='#FFF8DC')
        header_label.pack(pady=(10, 30))

        self.add_credit_row(self.credits_frame, "Danışman: Dr. İrfan KÖSESOY", "https://www.linkedin.com/in/irfankosesoy/")
        self.add_credit_row(self.credits_frame, "Proje Yürütücüsü: Faruk SİNER", "https://www.linkedin.com/in/faruksiner/")
        self.add_credit_row(self.credits_frame, "Proje Araştırmacısı: Emirhan Vedat KIVANÇ", "https://www.linkedin.com/in/emirhan-k%C4%B1van%C3%A7-b476b4219/")

        close_button_style = {'bg': '#FF6347', 'fg': 'white', 'font': ('Helvetica', 12, 'bold'), 'relief': 'raised', 'bd': 5}
        close_button = Button(self.credits_frame, text="X", command=self.close_credits, **close_button_style)
        close_button.pack(side=tk.BOTTOM, pady=20)

    def add_credit_row(self, frame, text, url):
        row_frame = Frame(frame, bg='#FFF8DC')
        row_frame.pack(pady=10)
        label = Label(row_frame, text=text, font=('Helvetica', 12), bg='#FFF8DC')
        label.pack(side=tk.LEFT, padx=10)

        linkedin_icon_path = 'src/linkedin.png'
        linkedin_icon = ImageTk.PhotoImage(Image.open(linkedin_icon_path).resize((20, 20), Image.Resampling.LANCZOS))
        link_button = Button(row_frame, image=linkedin_icon, command=lambda url=url: webbrowser.open(url), relief=tk.FLAT, bg='#FFF8DC')
        link_button.image = linkedin_icon  # Keep a reference to avoid garbage collection
        link_button.pack(side=tk.RIGHT)

    def close_credits(self):
        self.credits_frame.destroy()
        self.show_main_ui()

    def hide_main_ui(self):
        self.image_frame.pack_forget()
        self.sidebar.pack_forget()

    def show_main_ui(self):
        self.image_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.sidebar.pack(side=tk.RIGHT, fill=tk.Y)

    def upload_image(self):
        file_paths = filedialog.askopenfilenames(filetypes=[("Image files", "*.jpg *.jpeg *.png *.tiff *.bmp *.gif")])
        for file_path in file_paths:
            if file_path not in self.image_paths:
                self.image_paths.append(file_path)
                self.raw_images[file_path] = cv2.imread(file_path)

                img = self.raw_images[file_path]
                height, width, _ = img.shape
                if height % 512 == 0 and width % 512 == 0:
                    self.run_prediction(file_path)
                    self.add_thumbnail(file_path)
                else:
                    self.process_and_split_image(file_path)

                if self.current_image_path is None:
                    self.current_image_path = file_path
                    self.display_image(self.predicted_images[file_path])
                    self.update_building_count(file_path)

                self.switch_button.config(state=tk.NORMAL)
                self.zoom_in_button.config(state=tk.NORMAL)
                self.zoom_out_button.config(state=tk.NORMAL)

    def process_and_split_image(self, image_path):
        img = self.raw_images[image_path]
        height, width, _ = img.shape
        new_height = (height // 512 + 1) * 512
        new_width = (width // 512 + 1) * 512
        resized_img = cv2.resize(img, (new_width, new_height))

        base_filename = os.path.splitext(os.path.basename(image_path))[0]
        self.split_images_map[image_path] = []
        for i in range(new_height // 512):
            for j in range(new_width // 512):
                sub_img = resized_img[i * 512:(i + 1) * 512, j * 512:(j + 1) * 512]
                sub_image_path = os.path.join('splits', f"{base_filename}_{i}_{j}.png")
                os.makedirs('splits', exist_ok=True)
                cv2.imwrite(sub_image_path, sub_img)
                self.raw_images[sub_image_path] = sub_img
                self.split_images_map[image_path].append(sub_image_path)
                self.run_prediction(sub_image_path)
                self.add_thumbnail(sub_image_path)

    def add_thumbnail(self, file_path):
        thumbnail_text = os.path.basename(file_path)
        self.thumbnail_listbox.insert(tk.END, f"🖼️ {thumbnail_text}")

    def run_prediction(self, image_path):
        save_dir = 'output'
        os.makedirs(save_dir, exist_ok=True)
        results = self.model.predict(source=image_path, save=False, show=False)
        self.plot_predictions(results, save_dir)
        self.predicted_images[image_path] = cv2.imread(os.path.join(save_dir, os.path.basename(image_path)))

    def plot_predictions(self, results, save_dir):
        for result in results:
            img = result.orig_img  # Original image
            boxes = result.boxes  # Detected bounding boxes

            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw bounding box

            img_filename = os.path.join(save_dir, os.path.basename(result.path))
            cv2.imwrite(img_filename, img)

    def display_image(self, img):
        if isinstance(img, Image.Image):
            img_tk = ImageTk.PhotoImage(img)
        else:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            img_tk = ImageTk.PhotoImage(img_pil)
        self.canvas.img_tk = img_tk  # Keep reference to avoid garbage collection
        self.canvas.delete("all")
        self.canvas.create_image(self.canvas.winfo_width() // 2, self.canvas.winfo_height() // 2, anchor=tk.CENTER, image=img_tk)

    def switch_image(self):
        if self.current_image_path:
            if self.showing_raw:
                self.display_image(self.predicted_images[self.current_image_path])
                self.switch_button.config(text="Show Raw Image")
            else:
                self.display_image(self.raw_images[self.current_image_path])
                self.switch_button.config(text="Show Predicted Image")
            self.showing_raw = not self.showing_raw
            self.apply_zoom()

    def zoom_in(self):
        if self.current_image_path and self.zoom_scale < self.max_zoom:
            self.zoom_scale += 0.1
            self.apply_zoom()

    def zoom_out(self):
        if self.current_image_path and self.zoom_scale > self.min_zoom:
            self.zoom_scale -= 0.1
            self.apply_zoom()

    def apply_zoom(self):
        img = self.raw_images[self.current_image_path] if self.showing_raw else self.predicted_images[self.current_image_path]
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        width, height = img_pil.size
        zoomed_img = img_pil.resize((int(width * self.zoom_scale), int(height * self.zoom_scale)), Image.Resampling.LANCZOS)
        self.display_image(zoomed_img)

    def on_mouse_wheel(self, event):
        if event.delta > 0:
            self.zoom_in()
        else:
            self.zoom_out()

    def pan_start(self, event):
        self.canvas.scan_mark(event.x, event.y)

    def pan_move(self, event):
        self.canvas.scan_dragto(event.x, event.y, gain=1)

    def on_thumbnail_double_click(self, event):
        selection = self.thumbnail_listbox.curselection()
        if selection:
            index = selection[0]
            selected_image_name = self.thumbnail_listbox.get(index).split("🖼️ ")[1]  # Get the file name
            selected_image_path = [key for key in self.raw_images.keys() if selected_image_name in key][0]
            self.current_image_path = selected_image_path
            self.display_image(self.predicted_images[selected_image_path])
            self.update_building_count(selected_image_path)
            self.showing_raw = False
            self.switch_button.config(text="Show Raw Image")
            self.zoom_scale = 1.0  # Reset zoom when switching images
            self.apply_zoom()

    def update_building_count(self, image_path):
        count = self.count_buildings(image_path)
        self.building_count_label.config(text=f"Buildings Detected: {count}")

    def count_buildings(self, image_path):
        result = self.model.predict(source=image_path, save=False, show=False)[0]
        return len(result.boxes)

if __name__ == "__main__":
    root = tk.Tk()
    app = YoloApp(root)
    root.mainloop()
