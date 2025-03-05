import open3d as o3d
import open3d.visualization.gui as gui

def main():
    app = gui.Application.instance
    app.initialize()
    window = app.create_window("Test Window", 800, 600)
    app.run()  # ここでウィンドウが表示され続けるはず

if __name__ == "__main__":
    main()
