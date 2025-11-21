#include "fast_cd_viewer.h"
#include "fast_cd_viewer_vertex_selector.h"
#include "fast_cd_viewer_custom_shader.h"
#include "fast_cd_viewer_parameters.h"

#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <pybind11/eigen.h>

namespace py = pybind11;

using EigenDStride = Stride<Eigen::Dynamic, Eigen::Dynamic>;
template <typename MatrixType>
using EigenDRef = Ref<MatrixType, 0, EigenDStride>;

void bind_viewer(py::module& m)
{
    py::class_<fast_cd_viewer>(m, "fast_cd_viewer")
        .def(py::init<>())

        .def("set_mesh", &fast_cd_viewer::set_mesh)
        .def("set_vertices", &fast_cd_viewer::set_vertices)
        .def("invert_normals", &fast_cd_viewer::invert_normals)
        .def("set_camera_center", &fast_cd_viewer::set_camera_center)
        .def("set_camera_eye", &fast_cd_viewer::set_camera_eye)
        .def("set_camera_zoom", &fast_cd_viewer::set_camera_zoom)
        .def("clear", &fast_cd_viewer::clear)
        .def("compute_normals", &fast_cd_viewer::compute_normals)

        // --------------------
        // FIXED add_mesh #
        // --------------------
        .def("add_mesh", [](fast_cd_viewer& v) {
            int id = -1;
            v.add_mesh(id);
            return id;
        })

        .def("add_mesh", [](fast_cd_viewer& v,
                            EigenDRef<MatrixXd> V,
                            EigenDRef<MatrixXi> F) {
            int id = -1;
            v.add_mesh(V, F, id);
            return id;
        })

        // -------------------------
        // FIXED CALLBACK WRAPPERS #
        // -------------------------
        .def("set_pre_draw_callback",
            [](fast_cd_viewer& v, std::function<void(void)> func)
            {
                v.igl_v->callback_pre_draw =
                    [func](igl::opengl::glfw::Viewer&) -> bool {
                        func();
                        return false;
                    };
            })

        .def("set_key_callback",
            [](fast_cd_viewer& v,
               std::function<bool(unsigned int, int)> func)
            {
                v.igl_v->callback_key_pressed =
                    [func](igl::opengl::glfw::Viewer&, unsigned int key, int mod) -> bool {
                        return func(key, mod);
                    };
            })

        .def("set_face_based", &fast_cd_viewer::set_face_based)
        .def("set_color",
            static_cast<void (fast_cd_viewer::*)(const RowVector3d&, int)>
            (&fast_cd_viewer::set_color))

        .def("set_background_color",
            [](fast_cd_viewer& v, const Eigen::RowVector3d& color) {
                v.set_background_color(color.transpose());
            },
            "Set the background color (RGB values 0-1)")

        .def("set_lighting_factor", &fast_cd_viewer::set_lighting_factor)
        .def("set_show_lines", &fast_cd_viewer::set_show_lines)
        .def("get_show_lines", &fast_cd_viewer::get_show_lines)
        .def("set_show_faces", &fast_cd_viewer::set_show_faces)
        .def("get_show_faces", &fast_cd_viewer::get_show_faces)

        .def("launch", &fast_cd_viewer::launch)

        // ----------------------
        // FIXED GUIMZO BINDING #
        // ----------------------
        .def("init_guizmo",
            [](fast_cd_viewer& v,
               bool visible,
               EigenDRef<Matrix4f> A0,
               std::function<void(const Matrix4f&)> func,
               std::string op)
            {
                v.guizmo->visible = visible;
                v.guizmo->T = A0;

                v.guizmo->callback =
                    [func](const Matrix4f& A) {
                        func(A);
                    };

                if (op == "scale")      v.guizmo->operation = ImGuizmo::SCALE;
                else if (op == "translate") v.guizmo->operation = ImGuizmo::TRANSLATE;
                else if (op == "rotate")    v.guizmo->operation = ImGuizmo::ROTATE;
            })

        .def("change_guizmo_op",
            [](fast_cd_viewer& v, const std::string& op)
            {
                if (op == "scale")      v.guizmo->operation = ImGuizmo::SCALE;
                else if (op == "translate") v.guizmo->operation = ImGuizmo::TRANSLATE;
                else if (op == "rotate")    v.guizmo->operation = ImGuizmo::ROTATE;
            })

        .def("set_texture", &fast_cd_viewer::set_texture,
             " const string & tex_png, const MatrixXd & TC, const MatrixXi & FTC, int id ");

    // --------------------------------------------------
    // Vertex selector binding
    // --------------------------------------------------
    py::class_<fast_cd_viewer_vertex_selector, fast_cd_viewer>(m, "fast_cd_viewer_vertex_selector")
        .def(py::init<>())

        .def("query_new_handles",
            [](fast_cd_viewer_vertex_selector& v)
            {
                MatrixXd C; VectorXi CI;
                bool added = v.query_new_handles(C, CI);
                return std::make_tuple(C, CI, added);
            })

        .def("query_new_handles_on_mesh",
            [](fast_cd_viewer_vertex_selector& v,
               EigenDRef<MatrixXd> V,
               EigenDRef<MatrixXi> T)
            {
                MatrixXd C; VectorXi CI;
                bool added = v.query_new_handles_on_mesh(C, CI, V, T);
                return std::make_tuple(C, CI, added);
            });

    // --------------------------------------------------
    // Custom shader binding
    // --------------------------------------------------
    py::class_<fast_cd_viewer_custom_shader, fast_cd_viewer>(m, "fast_cd_viewer_custom_shader")
        .def(py::init<>())

        .def(py::init<std::string&, std::string&, int, int>(),
            py::arg("vertex_shader"),
            py::arg("fragment_shader"),
            py::arg("max_num_primary_bones"),
            py::arg("max_num_secondary_bones"))

        .def("launch", &fast_cd_viewer_custom_shader::launch,
             py::arg("max_fps"), py::arg("render"))

        .def("init_buffers", &fast_cd_viewer_custom_shader::init_buffers)
        .def("free_buffers", &fast_cd_viewer_custom_shader::free_buffers)

        .def("set_primary_weights", &fast_cd_viewer_custom_shader::set_primary_weights)
        .def("set_secondary_weights", &fast_cd_viewer_custom_shader::set_secondary_weights)
        .def("set_weights", &fast_cd_viewer_custom_shader::set_weights)

        .def("set_primary_bone_transforms", &fast_cd_viewer_custom_shader::set_primary_bone_transforms)
        .def("set_secondary_bone_transforms", &fast_cd_viewer_custom_shader::set_secondary_bone_transforms)
        .def("set_bone_transforms", &fast_cd_viewer_custom_shader::set_bone_transforms)

        .def("set_caustics_atlas", &fast_cd_viewer_custom_shader::set_caustics_atlas)

        .def("set_uniform",
            [](fast_cd_viewer_custom_shader& v, const std::string& name, int value, int id) {
                v.set_uniform(name, value, id);
            },
            py::arg("uniform_name"), py::arg("value"), py::arg("id") = 0)

        .def("set_uniform",
            [](fast_cd_viewer_custom_shader& v, const std::string& name, float value, int id) {
                v.set_uniform(name, value, id);
            },
            py::arg("uniform_name"), py::arg("value"), py::arg("id") = 0)

        .def("updateGL", &fast_cd_viewer_custom_shader::updateGL);

    // --------------------------------------------------
    // Parameter class binding
    // --------------------------------------------------
    py::class_<fast_cd_viewer_parameters>(m, "fast_cd_viewer_parameters")
        .def(py::init<>())

        .def("set_texture",
            static_cast<void (fast_cd_viewer_parameters::*)
                (std::string, std::string, double, RowVector3d)>
                (&fast_cd_viewer_parameters::set_texture))

        .def("set_texture",
            static_cast<void (fast_cd_viewer_parameters::*)
                (std::string, std::string)>
                (&fast_cd_viewer_parameters::set_texture));
}
